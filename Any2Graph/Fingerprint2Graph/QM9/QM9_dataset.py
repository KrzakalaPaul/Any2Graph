from chainer_chemistry import datasets
from chainer_chemistry.dataset.preprocessors.ggnn_preprocessor import GGNNPreprocessor
from torch.utils.data import Dataset
import torch 
from torch.nn.functional import one_hot
from torchtext.functional import add_token
import numpy as np
import networkx as nx
import pandas as pd 
import os 
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import GetHashedMorganFingerprint,GetMACCSKeysFingerprint
from torchtext.vocab import vocab
from rdkit.Chem import  MolFromSmiles, rdmolops

def ECFP4(mol, return_bits=False, **kwargs):
    fp = GetHashedMorganFingerprint(mol, radius=2, nBits=2048)
    return list(fp.GetNonzeroElements()) if return_bits else fp
    
class QM9_Dataset(Dataset):
    def __init__(self,split,root_path = 'Any2Graph/Fingerprint2Graph/QM9/data/',dataset_size=-1):
        
        super(QM9_Dataset).__init__()
        
        path = os.path.join(root_path,'QM9_'+split+'_smiles.csv')
        self.smiles = pd.read_csv(path)
            
        # TRONCATE DATASET FOR EXP
        if dataset_size>0:
            self.smiles = self.smiles.iloc[:dataset_size]
        
        self.vocab = vocab({str(i):i for i in range(2048)})
        self.vocab.append_token("<unk>")
        self.vocab.append_token("<sos>")
        self.vocab.set_default_index(self.vocab["<unk>"])
        self.text_pipeline = lambda x: add_token(self.vocab(x),token_id=self.vocab["<sos>"],begin=True)
        self.node_atom_map = {0: "C", 1: "N", 2: "O", 3: "F"}
        
        self.colors =  np.array([[0. ,0. ,0. ],
                                 [1/5,1/4,1/2],
                                 [3/4,1/4,1/4],
                                 [1/4,3/4,1/4]])
        
    def __len__(self):
        return len(self.smiles)
    
    def symbol_to_one_hot(self,symbol):
        tensor = torch.zeros(4)
        if symbol == 'C':
            tensor[0] = 1
        if symbol == 'N':
            tensor[1] = 1
        if symbol == 'O':
            tensor[2] = 1
        if symbol == 'F':
            tensor[3] = 1
        return tensor
    
    def __getitem__(self, idx):
        
        smile = self.smiles.iloc[idx].values[0]
        mol = MolFromSmiles(smile)
        
        A = torch.tensor(rdmolops.GetAdjacencyMatrix(mol)).to(torch.float32)
        F = torch.stack([self.symbol_to_one_hot(atom.GetSymbol()) for atom in mol.GetAtoms()]).to(torch.float32)
        graph = {'A':A,'F':F}
        
        fp = ECFP4(mol,return_bits=True)
        fp = [str(bit) for bit in fp]
        tokens = self.text_pipeline(fp)
        
        return tokens,graph,idx
    
    def plot_input(self,index,ax,frame=False,fontsize=12):
        
        x,_,_= self[index]

        text = ''
        for k,u in enumerate(x):
            if k%3 == 2:
                text += '\n'
            text += str(u)+' '
            
        ax.text(0.5,0.5,text, wrap=True, ha='center', va='center',fontsize=fontsize)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if frame:
            pass
        else:
            ax.axis('off')

        
    def plot_trgt(self,index_trgt,ax_trgt,frame=False):
        
        _,graph_trgt_dic,_= self[index_trgt]
        
        A_trgt = graph_trgt_dic['A'].detach().cpu().numpy()
        F_trgt = graph_trgt_dic['F'].detach().cpu().numpy()
        
        graph_trgt_nx = nx.from_numpy_array(A_trgt)
        labels = {}
        color_map = []
        for k, f in enumerate(F_trgt):
            labels[k] = self.node_atom_map[np.argmax(f)]
            color_map.append(self.colors[np.argmax(f)])

        pos = nx.kamada_kawai_layout(graph_trgt_nx)
        
        nx.draw_networkx_nodes(graph_trgt_nx,node_color="k",ax=ax_trgt, pos=pos)
        nx.draw_networkx_nodes(graph_trgt_nx, pos, node_size=200, node_color=color_map,ax=ax_trgt,alpha=1)
        [nx.draw_networkx_edges(graph_trgt_nx,pos=pos,edgelist=[(u,v)],alpha=1,width=2,ax=ax_trgt) for u,v in graph_trgt_nx.edges] #loop through edges and draw the
        
        ax_trgt.set_xlim((-1.1,1.1))
        ax_trgt.set_ylim((-1.1,1.1))
        ax_trgt.set_xticklabels([])
        ax_trgt.set_yticklabels([])
        
        if frame:
            pass
        else:
            ax_trgt.axis('off')
    
    def plot_pred(self,F,A,index_trgt,ax_pred,frame=False):

        _,graph_trgt_dic,_= self[index_trgt]
        
        A_trgt = graph_trgt_dic['A'].detach().cpu().numpy()
        F_trgt = graph_trgt_dic['F'].detach().cpu().numpy()
        
        graph_trgt_nx = nx.from_numpy_array(A_trgt)
        labels = {}
        color_map = []
        for k, f in enumerate(F_trgt):
            labels[k] = self.node_atom_map[np.argmax(f)]
            color_map.append(self.colors[np.argmax(f)])

        pos = nx.kamada_kawai_layout(graph_trgt_nx)

        graph_pred = nx.from_numpy_array(A)
        color_map = []
        for k, f in enumerate(F):
            color_map.append(self.colors[np.argmax(f)])
        
        nx.draw_networkx_nodes(graph_pred,node_color="k",ax=ax_pred, pos=pos)
        nx.draw_networkx_nodes(graph_pred, pos, node_size=200, node_color=color_map,ax=ax_pred,alpha=1)
        [nx.draw_networkx_edges(graph_pred,pos=pos,edgelist=[(u,v)],alpha=graph_pred[u][v]["weight"],width=2,ax=ax_pred) for u,v in graph_pred.edges] #loop through edges and draw the
        ax_pred.set_xlim(-1.1,1.1)
        ax_pred.set_ylim(-1.1,1.1)
        if frame:
            pass
        else:
            ax_pred.axis('off')

