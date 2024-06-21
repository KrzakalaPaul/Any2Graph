import torch.nn as nn
import torch
from numpy import pi


class LearnablePositonnalEncodings(nn.Module):

    def __init__(self,feature_dim,H,W):
        super().__init__()

        self.H = H
        self.W = W
        self.row_embed = nn.Parameter(torch.rand(H, feature_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(W, feature_dim // 2))
        nn.init.uniform_(self.row_embed)
        nn.init.uniform_(self.col_embed)

    def forward(self,x):
        positionnal_embeddings = torch.cat([self.row_embed.unsqueeze(1).repeat(1,self.W,1), self.col_embed.unsqueeze(0).repeat(self.H,1,1)],dim=2)
        positionnal_embeddings = torch.permute(positionnal_embeddings,(2,0,1))
        #return positionnal_embeddings[None,:,:,:]
        B = len(x)
        return positionnal_embeddings.unsqueeze(0).repeat(B,1,1,1)
    

class SinePositonnalEncodings(nn.Module):

    def __init__(self,feature_dim,H,W):
        super().__init__()

        self.H = H
        self.W = W

        dim_t = torch.arange(feature_dim//2, dtype=torch.float32)
        dim_t = 10000 ** (2 * (dim_t // 2) / (feature_dim//2))
        pos_x = 2*pi*torch.arange(H, dtype=torch.float32)/H
        pos_y = 2*pi*torch.arange(W, dtype=torch.float32)/W

        pos_x = pos_x[:, None]/dim_t
        pos_y = pos_y[:, None]/dim_t

        self.row_embed = torch.concat((pos_x[:, 0::2].sin(), pos_x[:,1::2].cos()), dim=1)
        self.col_embed = torch.concat((pos_y[:, 0::2].sin(), pos_y[:,1::2].cos()), dim=1)

    def forward(self,x):
        positionnal_embeddings = torch.cat([self.row_embed.unsqueeze(1).repeat(1,self.W,1), self.col_embed.unsqueeze(0).repeat(self.H,1,1)],dim=2)
        positionnal_embeddings = torch.permute(positionnal_embeddings,(2,0,1))
        #return positionnal_embeddings[None,:,:,:].to(x.device)
        B = len(x)
        return positionnal_embeddings.unsqueeze(0).repeat(B,1,1,1).to(x.device)
    
class NoPositonnalEncodings(nn.Module):
    def forward(self,x):
        return torch.zeros_like(x)


from torchvision.models.resnet import BasicBlock,ResNet

def troncated_resnet(C,remove_blocks):
    
    resnet_start = [nn.Conv2d(C,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False),
                    nn.BatchNorm2d(64,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True),
                    nn.ReLU()
                    ]
    resnet18 = ResNet(BasicBlock, [2, 2, 2, 2])
    resnet_middle = list(resnet18.children())[4:-2-remove_blocks]
    
    model = nn.Sequential(*(resnet_start + resnet_middle ))  # Remove maxpool + 3 and 4th blocks + final linear layers
    return model
    
    
class EncoderTOULOUSE(nn.Module):

    def __init__(self, model_dim = 64,positionnal_encodings = 'learned'):
        super().__init__()

        input_shape = (1,64,64)

        self.params = {'model_dim':model_dim,
                       'input_shape':input_shape,
                       'positionnal_encodings':positionnal_encodings}

        self.cnn = nn.Sequential(troncated_resnet(input_shape[0],remove_blocks=1),
                                 nn.ReLU(),
                                 nn.Conv2d(256, model_dim, 1)) 
        
        input = torch.zeros((1,input_shape[0],input_shape[1],input_shape[2]))
        output_shape = self.cnn(input).shape
        print(output_shape)
        
        H,W = output_shape[-2:]

        if positionnal_encodings == 'learned':
            self.positionnal_encodings = LearnablePositonnalEncodings(model_dim,H,W)
        elif positionnal_encodings == 'fixed':
            self.positionnal_encodings = SinePositonnalEncodings(model_dim,H,W)
        elif positionnal_encodings == None:
            self.positionnal_encodings = nn.Identity()
        else:
            raise ValueError()


    def forward(self, x):

        x = self.cnn(x)
        pos_embed = self.positionnal_encodings(x)
        x = torch.flatten(x,start_dim=2)
        pos_embed = torch.flatten(pos_embed,start_dim=2)
        x = torch.transpose(x,dim0=1,dim1=2) # (Batchsize,NumberOfFeatures,FeatureDim)
        pos_embed = torch.transpose(pos_embed,dim0=1,dim1=2)
        mask = None
        return x,mask,pos_embed

class EncoderUSCities(nn.Module):

    def __init__(self, model_dim = 64, positionnal_encodings = 'learned'):
        super().__init__()

        input_shape = (1,128,128)

        self.params = {'model_dim':model_dim,
                       'input_shape':input_shape,
                       'positionnal_encodings':positionnal_encodings}

        self.cnn = nn.Sequential(troncated_resnet(input_shape[0],remove_blocks=0),
                                 nn.ReLU(),
                                 nn.Conv2d(512, model_dim, 1)) 

        input = torch.zeros((1,input_shape[0],input_shape[1],input_shape[2]))
        output_shape = self.cnn(input).shape
        print(output_shape)
        
        H,W = output_shape[-2:]

        if positionnal_encodings == 'learned':
            self.positionnal_encodings = LearnablePositonnalEncodings(model_dim,H,W)
        elif positionnal_encodings == 'fixed':
            self.positionnal_encodings = SinePositonnalEncodings(model_dim,H,W)
        elif positionnal_encodings == None:
            self.positionnal_encodings = nn.Identity()
        else:
            raise ValueError()


    def forward(self, x):

        x = self.cnn(x)
        pos_embed = self.positionnal_encodings(x)
        x = torch.flatten(x,start_dim=2)
        pos_embed = torch.flatten(pos_embed,start_dim=2)
        x = torch.transpose(x,dim0=1,dim1=2) # (Batchsize,NumberOfFeatures,FeatureDim)
        pos_embed = torch.transpose(pos_embed,dim0=1,dim1=2)
        mask = None
        return x,mask,pos_embed
    

    
    
