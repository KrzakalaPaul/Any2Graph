from Any2Graph.base_encoder_class import Encoder
import torch
import torch.nn as nn
    
from torchvision.models.resnet import BasicBlock,ResNet

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
    
from numpy import pi

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


def get_troncated_resnet(number_of_blocks_to_remove = 2):
    resnet18 = ResNet(BasicBlock, [2, 2, 2, 2])
    model = nn.Sequential(*(list(resnet18.children())[:3] + list(resnet18.children())[4:-2 - number_of_blocks_to_remove]))  # Remove maxpool + 3 and 4th blocks + final linear layers
    return model

class TroncatedResNet(nn.Module):

    def __init__(self, model_dim = 64, input_shape = (3,32,32),positionnal_encodings = 'learned'):
        super().__init__()

        self.cnn = nn.Sequential(get_troncated_resnet(),
                                 nn.ReLU(),
                                 nn.Conv2d(128, model_dim, 1)) 

        input = torch.zeros((64,input_shape[0],input_shape[1],input_shape[2]))
        output_shape = self.cnn(input).shape
        print(f'Output Shape of the CNN: {output_shape}')

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