import scipy
import os
import sys
import numpy as np
import random
import pickle
import json
import scipy.ndimage
from PIL import Image
import math
import torch
import pyvista
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
import torchvision.transforms.functional as tvf
from sklearn.model_selection import train_test_split
import networkx as nx
from  torchvision.transforms.functional import rotate,hflip
from random import choice
from math import cos,sin,pi
import time


def graph_from_file(file):
    
    vtk_data = pyvista.read(file)
        
    F = torch.tensor(np.float32(np.asarray(vtk_data.points)), dtype=torch.float)[:,:2]
    edges = torch.tensor(np.asarray(vtk_data.lines.reshape(-1, 3)), dtype=torch.int64)[:,1:]
    A = csr_matrix((np.ones(len(edges)), (edges[:,0], edges[:,1])), shape=(len(F), len(F)))
    A = A + A.T
    A = torch.tensor(A.todense(), dtype=torch.float)
    h = torch.ones(len(F), dtype=torch.float)
    
    m = len(vtk_data.points)
    
    return {'F': F, 'A': A, 'h': h}, m

class USCities_Dataset(Dataset):
    r"""
    Generates a subclass of the PyTorch torch.utils.data.Dataset class
    """
    
    def __init__(self, root_path='Any2Graph/Sat2Graph/USCities/data/', split="valid",augment_data=False,rgb=False,Mmax=17,dataset_size=-1):
        r"""
        """
        
        if split == "train" or split == "valid":
            
            img_folder = os.path.join(root_path,'20cities/train_data/raw')
            vtk_folder = os.path.join(root_path,'20cities/train_data/vtp')
            seg_folder = os.path.join(root_path,'20cities/train_data/seg')
            
            all_files = os.listdir(img_folder)
            train_indices, test_indices = train_test_split(all_files, test_size=0.05, random_state=42)

            split_files = train_indices if split == "train" else test_indices
  
        
        if split == "test":
            
            img_folder = os.path.join(root_path,'20cities/test_data/raw')
            vtk_folder = os.path.join(root_path,'20cities/test_data/vtp')
            seg_folder = os.path.join(root_path,'20cities/test_data/seg')

            split_files = os.listdir(img_folder)

        print('Loading data...')
        
        tic = time.time()
        
        self.graph = []
        self.img = []
        
        counter = 0
        for file_ in split_files:
            
            file_ = file_[:-8]
            
            img_file = os.path.join(img_folder, file_+'data.png')
            vtk_file = os.path.join(vtk_folder, file_+'graph.vtp')
            seg_file = os.path.join(seg_folder, file_+'seg.png')
            
            G, m = graph_from_file(vtk_file)
            
            if m <= Mmax:
                
                self.graph.append(G)
                
                if rgb:
                    self.img.append((255*tvf.to_tensor(Image.open(img_file).convert('RGB')).permute(2,1,0)).type(torch.uint8))
                else:
                    self.img.append(2*tvf.to_tensor(Image.open(seg_file).convert('L')).permute(2,1,0).type(torch.int8)-1)
                        
                counter += 1
                if counter == dataset_size:
                    break
                
        tac = time.time()
        print(f'Loading complete, took {tac-tic:.2f} seconds')
        print('Dataset size:',len(self.img))
        
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.augment = augment_data 
        self.rgb = rgb

    def __len__(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return len(self.img)
    
    def transform(self,x,y):
    
        flip = choice([True,False])
        angle = choice([0,90,180,270])
        
        F = y['F'].clone()

        if flip:
            x = hflip(x)
            F = F - 0.5 # centering
            F[:,0] = -F[:,0]
            F = F + 0.5 # uncentering
   
        x = rotate(x,angle)
        theta = pi*angle/180
        rot = torch.tensor([[cos(theta),sin(theta)],
                            [-sin(theta),cos(theta)]],dtype = F.dtype, device = F.device)
        F = F - 0.5 # centering
        F = F@rot.T
        F = F + 0.5 # uncentering
        
        z = {'A':y['A'], 'F':F}
        
        return x,z
    
    def __getitem__(self, idx):
        """[summary]

        Args:
            idx ([type]): [description]

        Returns:
            [type]: [description]
        """
        
        if self.rgb:
            image_data = self.img[idx].clone().float()
            image_data = image_data.permute(2,0,1)
            image_data = image_data/255.0
            image_data = tvf.normalize(image_data, mean=self.mean, std=self.std)
        else:
            image_data = self.img[idx].clone().float()
            image_data = image_data/torch.max(image_data)
            image_data = image_data.permute(2,0,1)
        
        input = image_data
        target = self.graph[idx]
        
        # TO AVOID WEIRD COPY ISSUES
        target = {'A':target['A'].clone(),'F':target['F'].clone(),'h':target['h'].clone()}
        
        if self.augment:
            input, target = self.transform(input, target)
        
        return input, target, idx
    
    def plot_img(self,index,ax):

        if self.rgb:
            image_data = self.img[index].float()
            image_data = image_data
            image_data = image_data/255.0
        else:
            image_data = self.img[index].float()
            image_data = image_data/torch.max(image_data)
            image_data = image_data
        ax.imshow(1-image_data,cmap='Greys',vmin=0,vmax=1)
        ax.invert_yaxis()
        ax.set_axis_off()
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    
    def plot_trgt(self,index_trgt,ax_trgt,frame=False):
        
        _, graph_trgt_dic, _ = self.__getitem__(index_trgt)
        graph_trgt_nx = nx.from_numpy_array(graph_trgt_dic['A'].detach().cpu().numpy())

        pos = [f for f in graph_trgt_dic['F'].detach().cpu().numpy()]
        
        nx.draw_networkx_nodes(graph_trgt_nx,node_color="k",ax=ax_trgt, pos=pos)
        nx.draw_networkx_nodes(graph_trgt_nx, pos, node_size=200, node_color='#3182bd',ax=ax_trgt,alpha=0.9)
        [nx.draw_networkx_edges(graph_trgt_nx,pos=pos,edgelist=[(u,v)],alpha=1,width=3,ax=ax_trgt) for u,v in graph_trgt_nx.edges]
        
        ax_trgt.axis('equal')
        if frame:
            pass
        else:
            ax_trgt.axis('off')
    
    def plot_pred(self,F,A,index_trgt,ax_pred,frame=False):

        _, graph_trgt_dic, _ = self.__getitem__(index_trgt)
        pos = [f for f in graph_trgt_dic['F'].detach().cpu().numpy()]
            
        pos = [f for f in F]
        graph_pred = nx.from_numpy_array(A)

        nx.draw_networkx_nodes(graph_pred,node_color="k",ax=ax_pred, pos=pos)
        nx.draw_networkx_nodes(graph_pred, pos, node_size=200, node_color='#3182bd',ax=ax_pred,alpha=0.9)
        [nx.draw_networkx_edges(graph_pred,pos=pos,edgelist=[(u,v)],alpha=graph_pred[u][v]["weight"],width=3,ax=ax_pred) for u,v in graph_pred.edges]
        
        ax_pred.axis('equal')
        if frame:
            pass
        else:
            ax_pred.axis('off')

    
