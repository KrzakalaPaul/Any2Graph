import numpy as np 
import networkx as nx
from random import choice
from scipy.ndimage import gaussian_filter
from torch_geometric.utils.convert import from_networkx,to_networkx
import torch
from math import sqrt
from copy import deepcopy
from random import choice

class ColoringSampler():

    def __init__(self,Mmin=2,
                      Mmax=10,
                      shape=(32,32),
                      sigma_filter=0.4,
                      sigma_noise=0.02,
                      node_weight_treshold=None,
                      edge_weight_treshold=None):
        self.Mmin = Mmin
        self.Mmax = Mmax
        self.colors =  np.array([[1/4,1/4,3/4],
                                 [1/4,3/4,1/4],
                                 [3/4,1/4,1/4],
                                 [1,4/5,2/5]])
        self.shape = shape
        self.sigma_filter = sigma_filter
        self.sigma_noise = sigma_noise
        self.node_weight_treshold = node_weight_treshold
        self.edge_weight_treshold = edge_weight_treshold
        
    def get_sample(self, torch_geometric = False):

        M = np.random.randint(self.Mmin,self.Mmax+1)
        K = len(self.colors)
        pixels_total = self.shape[0]*self.shape[1]
        edge_normalization = sqrt(self.shape[0]**2+self.shape[1]**2)
        centroids = np.random.uniform(0,1,(2,M))
        X = np.linspace(0,1,self.shape[0])
        Y = np.linspace(0,1,self.shape[1])
        dists = np.abs(X[:,None,None] - centroids[None,0,:]) + np.abs(Y[None,:,None] - centroids[1,None,:])
        closest = np.argmin(dists,axis=2)

        unique, counts = np.unique(closest, return_counts=True)

        if len(unique) != M:
            print('Discarding because empty node')
            return self.get_sample(torch_geometric = torch_geometric)

        graph = nx.Graph()
        graph.add_nodes_from([(m,{'centroid':centroids[:,m], 'h':counts[m]/pixels_total}) for m in range(M)])

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if j+1<self.shape[1]:
                    u = closest[i,j+1]
                    v = closest[i,j]
                    if u!=v:
                        if graph.has_edge(u,v):
                            graph[u][v]['edge_weight']+=1/edge_normalization
                        else:
                            graph.add_edge(u, v, edge_weight = 1/edge_normalization)
                        
                if i+1<self.shape[0]:
                    u = closest[i+1,j]
                    v = closest[i,j]
                    if u!=v:
                        if graph.has_edge(u,v):
                            graph[u][v]['edge_weight']+=1/edge_normalization
                        else:
                            graph.add_edge(u, v, edge_weight = 1/edge_normalization)

        if self.edge_weight_treshold == None:
            # if using a edge weight, we ensure it is in [0,1]
            for (u,v) in graph.edges:
                if graph[u][v]['edge_weight'] > 1:
                    print('Discarding because edge too big')
                    return self.get_sample(torch_geometric = torch_geometric)
                
        # Coloring the graph
        valid_coloring = False
        while not(valid_coloring):
            
            coloring = {}
            valid_coloring = True

            for m in range(M):

                valid_colors = set([k for k in range(K)])
                for n in graph.neighbors(m):
                    if n in coloring:
                        valid_colors.discard(coloring[n])
                
                if len(valid_colors) == 0:
                    valid_coloring = False
                    break
                else:
                    coloring[m] = choice(list(valid_colors))

        nx.set_node_attributes(graph, coloring, 'color')
        
        # Apply tresholds if any
        if self.edge_weight_treshold != None:
            for (u,v) in graph.edges:
                if graph[u][v]['edge_weight']*edge_normalization < self.edge_weight_treshold:
                    graph.remove_edge(u, v)
                else:
                    graph[u][v]['edge_weight'] = 1
                    
        if self.node_weight_treshold != None:             
            nodes_to_remove = []
            for u in graph.nodes:
                if graph.nodes[u]['h']*pixels_total < self.node_weight_treshold:
                    nodes_to_remove.append(u)
                else:
                    graph.nodes[u]['h'] = 1
            for u in nodes_to_remove:
                graph.remove_node(u)

        

        node_colors = np.vstack([self.colors[coloring[m]] for m in range(M)])
        img = np.take(node_colors,closest,axis=0)
        img = gaussian_filter(img,self.sigma_filter, mode = 'nearest')
        img += np.random.normal(0,self.sigma_noise,size=img.shape)
        img = np.clip(img,0,1)

        if torch_geometric:
            return img,self.to_torch_geometric(graph)
        return img,graph
    
    def to_torch_geometric(self,graph):
        
        graph = deepcopy(graph)
        centroids = nx.get_node_attributes(graph, "centroid")
        for m in graph.nodes:
            centroids[m] = torch.from_numpy(centroids[m])
        nx.set_node_attributes(graph,centroids,"centroid")
        graph = from_networkx(graph)
        K = len(self.colors)
        graph.F = torch.eye(K)[graph.color]
        return graph
    
    def from_torch_geometric(self,graph):
        graph = deepcopy(graph)

        return to_networkx(graph,node_attrs=['centroid','h','color'],edge_attrs=['edge_weight'],to_undirected=True)

if __name__ == '__main__':
    '''
    ### Plot Some graphs
    
    sampler = ColoringSampler(Mmin=10,
                              Mmax=15,
                              node_weight_treshold=3,
                              edge_weight_treshold=3)
    
    for _ in range(10):

        img,graph = sampler.get_sample()
        
        print(img.shape)
        print(graph.nodes(data=True))
        
        print(sampler.to_torch_geometric(graph).h)
        print(sampler.to_torch_geometric(graph).edge_weight)
        
        fig,(ax1,ax2) = plt.subplots(ncols=2)
        ax1.imshow(np.transpose(img,(1,0,2)),vmin=0,vmax=1,origin='lower')

        color_map = [sampler.colors[node[1]['color']] for node in graph.nodes(data=True)]
        pos = nx.get_node_attributes(graph, "centroid")
        edges_weights = [graph[u][v]['edge_weight'] for u,v in graph.edges]
        node_size = [500*node[1]['h'] for node in graph.nodes(data=True)]
        nx.draw(graph, node_color = color_map, ax = ax2, pos=pos, width = edges_weights, node_size=node_size)
        ax2.set_xlim(-0.1,1.1)
        ax2.set_ylim(-0.1,1.1)

        plt.show()
        '''
    ### Generate datasets
    

    '''
    import os
    
    name = 'coloring_small'
    sampler = ColoringSampler(Mmin=4,
                              Mmax=10,
                              node_weight_treshold=3,
                              edge_weight_treshold=3)
    
    if not os.path.exists('data/'+name):
        os.mkdir('data/'+name)
    

    for split in ['test','valid']:

        n_samples = 5000
        
        folder = 'data/'+name+'/'+split
        
        if not os.path.exists(folder):
            os.mkdir(folder)
            
        images = []
        graphs = []
                
        for k in range(n_samples):
            img,graph = sampler.get_sample(torch_geometric=True)
            img = img*255
            img = img.astype(np.uint8)
            images.append(img)
            graphs.append(graph)
        
        images = np.stack(images,0)
        np.save(folder+'/images.npy',images)
        torch.save(graphs,folder+'/graphs' )
    '''
    
    