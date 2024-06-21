import numpy as np 
import networkx as nx
from random import choice
from torch_geometric.utils.convert import to_networkx
import torch
import os
from torch_geometric.utils import to_dense_adj
from torchvision.transforms import RandomHorizontalFlip
from  torchvision.transforms.functional import rotate
from random import choice
from time import perf_counter
from torch.utils.data import Dataset

class ColoringDataset(Dataset):

    def __init__(self,root_path='Img2Graph/Coloring/data/',subset='small',split='test',augment_data=False,dataset_size=-1):
        
        super().__init__()
        
        root_path = os.path.join(root_path,subset,split)
        
        print('Loading Images...')
        tic = perf_counter()
        self.images = np.load(root_path+'/images.npy')
        tac = perf_counter()
        print(f'...images loaded, it took {tac-tic:.2f} seconds')
        print('Loading Graphs...')
        tic = perf_counter()
        self.graphs = torch.load(root_path+'/graphs')
        tac = perf_counter()
        print(f'...graphs loaded, it took {tac-tic:.2f}')
        
        # TRONCATE DATASET FOR EXP
        if dataset_size > 0:
            self.images = self.images[:dataset_size]
            self.graphs = self.graphs[:dataset_size]
            
        self.dataset_size = len(self.images)
   
        self.sizes = []
        for graph in self.graphs:
            self.sizes.append(graph.num_nodes)
            
        self.colors =  np.array([[1/4,1/4,3/4],
                                 [1/4,3/4,1/4],
                                 [3/4,1/4,1/4],
                                 [1,4/5,2/5]])
        
        self.augment_data = augment_data
        self.RandomFlip = RandomHorizontalFlip(p=0.5)
        
    def transform(self,img):
        img = self.RandomFlip(img)
        img = rotate(img,angle = choice([0,90,180,270]))
        return img
        
        
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32)/255
        x = torch.permute(torch.tensor(image, dtype=torch.float32),(2,0,1))
        if self.augment_data:
            x = self.transform(x)
        
        graph = self.graphs[idx]
        A = to_dense_adj(graph.edge_index,max_num_nodes=graph.num_nodes,edge_attr=graph.edge_weight).squeeze().to(torch.float32)
        F = graph.F.to(torch.float32)
        y = {'A':A,'F':F}
        
        return x,y,idx
    
    def plot_img(self,index,ax,frame=False):
        image = self.images[index].astype(np.float32)/255
        ax.imshow(np.transpose(image,(1,0,2)),vmin=0,vmax=1,origin='lower')
        if frame:
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.set_axis_off()
        
    def plot_trgt(self,index_trgt,ax_trgt,frame=False):
        
        graph_trgt_torch = self.graphs[index_trgt]
        graph_trgt_nx = to_networkx(graph_trgt_torch,edge_attrs=['edge_weight'],to_undirected=True)

        pos = [p.numpy() for p in graph_trgt_torch.centroid]
        color_map = [self.colors[c] for c in graph_trgt_torch.color] 

        nx.draw_networkx_nodes(graph_trgt_nx,node_color="k",ax=ax_trgt, pos=pos)
        nx.draw_networkx_nodes(graph_trgt_nx, pos, node_size=200, node_color=color_map,ax=ax_trgt,alpha=1)
        [nx.draw_networkx_edges(graph_trgt_nx,pos=pos,edgelist=[(u,v)],alpha=1,width=2,ax=ax_trgt) for u,v in graph_trgt_nx.edges] #loop through edges and draw the
            
        ax_trgt.axis('equal')
        if frame:
            pass
        else:
            ax_trgt.axis('off')
            
            
    def plot_pred(self,F,A,h,index_trgt,ax_pred,frame=False):

        graph_trgt_torch = self.graphs[index_trgt]
        pos = [p.numpy() for p in graph_trgt_torch.centroid]

        graph_pred = nx.from_numpy_array(A)
        color_map = [self.colors[f] for f in F]

        nx.draw_networkx_nodes(graph_pred,node_color="k",ax=ax_pred, pos=pos)
        nx.draw_networkx_nodes(graph_pred, pos, node_size=200, node_color=color_map,ax=ax_pred,alpha=1)
        [nx.draw_networkx_edges(graph_pred,pos=pos,edgelist=[(u,v)],alpha=graph_pred[u][v]["weight"],width=2,ax=ax_pred) for u,v in graph_pred.edges] #loop through edges and draw them
        
        ax_pred.axis('equal')
        if frame:
            pass
        else:
            ax_pred.axis('off')


