import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as tvf
from PIL import Image

import time
import pickle
import random

import numpy as np
import networkx as nx

from  torchvision.transforms.functional import rotate,hflip
from random import choice
from math import cos,sin,pi

class ToulouseRoadNetworkDataset(Dataset):
    r"""
    Generates a subclass of the PyTorch torch.utils.data.Dataset class
    """
    
    def __init__(self, root_path="dataset/", split="valid",
                 max_prev_node=4, step=0.001, use_raw_images=False, return_coordinates=False,augment=False,data_set_size=0):
        r"""
        
        :param root_path: root dataset path
        :param split: dataset split in {"train", "valid", "test", "augment"}
        :param max_prev_node: only return the last previous 'max_prev_node' elements in the adjacency row of a node
            default is 4, which corresponds to the 95th percentile in the dataset
        :param step: step size used in the dataset generation, default is 0.001° (around 110 metres per datapoint)
        :param use_raw_images: loads raw images if yes, otherwise faster and more compact numpy array representations
        :param return_coordinates: returns coordinates on the real map for each datapoint, used for qualitative studies
        """
        root_path += str(step)
        assert split in {"train", "valid", "test", "augment"}
        print(f"Started loading the dataset ({split})...")
        start_time = time.time()
        
        dataset_path = f"{root_path}/{split}.pickle"
        images_path = f"{root_path}/{split}_images.pickle"
        images_raw_path = f"{root_path}/{split}/images/"
        
        ids, (x_adj, x_coord, y_adj, y_coord, seq_len, map_coordinates) = load_dataset(dataset_path, max_prev_node,
                                                                                   return_coordinates)

        self.return_coordinates = return_coordinates
        self.ids = ["{:0>7d}".format(int(i)) for i in ids]
        self.x_adj = x_adj
        self.x_coord = x_coord
        self.graphs = LinearizedGraph_To_Graph(x_adj=x_adj,x_coord=x_coord)
        self.map_coordinates = map_coordinates
        self.augment = augment

        print(f"Started loading the images...")
        
        if use_raw_images:
            self.images = load_raw_images(ids, images_raw_path)
        else:
            self.images = load_images(ids, images_path)
            
        if data_set_size!=0:
            self.images = self.images[:data_set_size]
            self.graphs = self.graphs[:data_set_size]
            self.ids = self.ids[:data_set_size]
        
        print(f"Dataset loading completed, took {round(time.time() - start_time, 2)} seconds!")
        print(f"Dataset size: {len(self)}\n")

    def transform(self,x,y):
    
        flip = choice([True,False])
        angle = choice([0,90,180,270])
        
        F = y['F']
        if flip:
            x = hflip(x)
            F[:,0] = -F[:,0]

        x = rotate(x,angle)
        theta = - pi*angle/180
        rot = torch.tensor([[cos(theta),sin(theta)],
                            [-sin(theta),cos(theta)]],dtype = F.dtype, device = F.device)
        F = F@rot.T
        
        z = {'A':y['A'], 'F':F, 'h':y['h']}
        
        return x,z
        
    def __len__(self):
        r"""
        :return: dataset length
        """
        return len(self.ids)
    
    def __getitem__(self, idx):
        r"""
        :param idx: index in the dataset
        :return: chosen data point
        """
        x = torch.permute(self.images[idx],(2,0,1))
        y = self.graphs[idx]
        # TO AVOID WEIRD COPY ISSUES
        y = {'A':y['A'].clone(),'F':y['F'].clone()}
        if self.augment:
            x,y = self.transform(x,y)
        return x,y,idx
    
    def plot_img(self,index,ax):
        
        ax.imshow(self.images[index],cmap='Greys',vmin=-1,vmax=1,origin='lower')
        ax.set_axis_off()
        ax.invert_yaxis()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    
    def plot_trgt(self,index_trgt,ax_trgt,frame=False):
        
        graph_trgt_dic = self.graphs[index_trgt]
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

        
    def plot_pred_trgt(self,F,A,index_trgt,ax_trgt,ax_pred,frame=False):

        graph_trgt_dic = self.graphs[index_trgt]
        graph_trgt_nx = nx.from_numpy_array(graph_trgt_dic['A'].detach().cpu().numpy())

        pos = [f for f in graph_trgt_dic['F'].detach().cpu().numpy()]

        if ax_trgt!=None:
            print('not implemented')
            
        pos = [f for f in F]
        graph_pred = nx.from_numpy_array(A)
        
        #nx.draw(graph_pred, node_color = color_map, ax = ax_pred, pos=pos, width = edges_weights, node_size=node_size)
        nx.draw_networkx_nodes(graph_pred,node_color="k",ax=ax_pred, pos=pos)
        nx.draw_networkx_nodes(graph_pred, pos, node_size=200, node_color='#3182bd',ax=ax_pred,alpha=0.9)
        [nx.draw_networkx_edges(graph_pred,pos=pos,edgelist=[(u,v)],alpha=graph_pred[u][v]["weight"],width=3,ax=ax_pred) for u,v in graph_pred.edges] #loop through edges and draw them
        ax_pred.axis('equal')
        if frame:
            pass
        else:
            ax_pred.axis('off')



def LinearizedGraph_To_Graph(x_adj,x_coord):
    
    graphs = []

    for this_x_adj,this_x_coord in zip(x_adj,x_coord):
        
        A_linear = this_x_adj.numpy()
        A = decode_adj(A_linear)
        
        true_nodes = []
        for k in range(len(A)):
            if np.allclose(A[k],np.zeros(len(A))):
                pass
            else:
                true_nodes.append(k)
    
        A = torch.tensor(A[true_nodes][:,true_nodes]).to(torch.float32)
        F = this_x_coord[true_nodes].to(torch.float32)
        h = torch.ones(len(F)).to(torch.float32)
        
        graphs.append({'A':A,'F':F,'h':h,'G':None})
    
    return graphs




def decode_adj(adj_output):
    r"""
    Recover the adj matrix A from adj_output
    note: here adj_output has shape [N x max_prev_node], while A has shape [N x N]
    
    :param adj_output: outputs of the decoder
    :return: adjacency matrix A
    """
    '''
    
    '''

    max_prev_node = adj_output.shape[1]
    adj = np.zeros((adj_output.shape[0], adj_output.shape[0]))
    for i in range(adj_output.shape[0]):
        input_start = max(0, i - max_prev_node + 1)
        input_end = i + 1
        output_start = max_prev_node + max(0, i - max_prev_node + 1) - (i + 1)
        output_end = max_prev_node
        adj[i, input_start:input_end] = adj_output[i,::-1][output_start:output_end] # reverse order
    adj_full = np.zeros((adj_output.shape[0]+1, adj_output.shape[0]+1))
    n = adj_full.shape[0]
    adj_full[1:n, 0:n-1] = np.tril(adj, 0)
    adj_full = adj_full + adj_full.T

    return adj_full[1:, 1:]





def load_dataset(dataset_path, max_prev_node, return_coordinates):
    r"""
    Loads the chosen split of the dataset
    
    :param dataset_path: path of the dataset split pickle
    :param max_prev_node: only return the last previous 'max_prev_node' elements in the adjacency row of a node
    :param return_coordinates: returns coordinates on the real map for each datapoint
    :return:
    """
    with open(dataset_path, "rb") as pickled_file:
        dataset = pickle.load(pickled_file)
    
    list_x_adj = []
    list_x_coord = []
    list_y_adj = []
    list_y_coord = []
    list_seq_len = []
    list_original_xy = []
    ids = list(dataset.keys())
    random.Random(42).shuffle(ids)  # permute to remove any correlation between consecutive datapoints
    
    for id in ids:
        datapoint = dataset[id]
        
        x_adj = torch.FloatTensor(datapoint["bfs_nodes_adj"])
        x_coord = torch.FloatTensor(datapoint["bfs_nodes_points"])

        # add termination token (zero-vector) to model the termination of a connected component AND the whole graph
        x_adj = torch.cat([x_adj, torch.zeros_like(x_adj[0]).unsqueeze(0)])
        x_coord = torch.cat([x_coord, torch.zeros_like(x_coord[0]).unsqueeze(0)])
        # add 2nd termination token (zero-vector) to model the termination of a connected component AND the whole graph
        y_adj = torch.cat([x_adj[1:, :].clone(), torch.zeros_like(x_adj[0]).unsqueeze(0)])
        y_coord = torch.cat([x_coord[1:, :].clone(), torch.zeros_like(x_coord[0]).unsqueeze(0)])
        # slice up to max_prev_node length
        x_adj = x_adj[:, :max_prev_node]
        y_adj = y_adj[:, :max_prev_node]
        
        list_x_adj.append(x_adj)
        list_x_coord.append(x_coord)
        list_y_adj.append(y_adj)
        list_y_coord.append(y_coord)
        list_seq_len.append(len(x_adj))  # Seq len is computed here, after creating the actual sequence
        list_original_xy.append(datapoint["coordinates"])
    list_seq_len = torch.LongTensor(list_seq_len)
    
    if return_coordinates:
        return ids, (list_x_adj, list_x_coord, list_y_adj, list_y_coord, list_seq_len, list_original_xy)
    return ids, (list_x_adj, list_x_coord, list_y_adj, list_y_coord, list_seq_len, None)


def load_images(ids, images_path):
    r"""
    Load images from arrays in pickle files
    
    :param ids: ids of the images in the dataset order
    :param images_path: path of the pickle file
    :return: the images, as pytorch tensors
    """
    images = []
    with open(images_path, "rb") as pickled_file:
        images_features = pickle.load(pickled_file)
    for id in ids:
        img = torch.FloatTensor(images_features["{:0>7d}".format(int(id))])
        assert img.shape[1] == img.shape[2]
        assert img.shape[1] in {64}
        images.append(torch.permute(torch.tensor(img, dtype=torch.float32),(1,2,0)))
    
    return images


def load_raw_images(ids, images_path):
    r"""
    Load images from raw files
    
    :param ids: ids of the images in the dataset order
    :param images_path: path of the raw images
    :return: the images, as pytorch tensors
    """
    images = []
    for count, id in enumerate(ids):
        # if count % 10000 == 0:
        #     print(count)
        image_path = images_path + "{:0>7d}".format(int(id)) + ".png"
        img = Image.open(image_path).convert('L')
        img = tvf.to_tensor(img)
        assert img.shape[1] == img.shape[2]
        assert img.shape[1] in {64, 128}
        images.append(torch.permute(torch.tensor(img, dtype=torch.float32),(1,2,0)))
    return images



def normalize(x, normalization=True):
    r"""
    Image normalization in [-1,+1]
    
    :param x: input tensor
    :param normalization: if False, return the input
    :return:
    """
    if normalization:
        x = (x * 2) - 1
    return x


def denormalize(x, normalization=True):
    r"""
    Image denormalization, converting back to [0,+1]
    
    :param x: input tensor
    :param normalization: if False, return the input
    :return:
    """
    if normalization:
        x = (x + 1) / 2
    return x


if __name__ == "__main__":
    dataset = ToulouseRoadNetworkDataset(root_path='data/',split="train", step=0.001, max_prev_node=4, use_raw_images=False)
    
    start_time = time.time()
    for d in dataset:
        x,y,idx = d
    print(f"Iteration over the dataset completed, took {round(time.time() - start_time, 2)}s!")