from torch_geometric.datasets import Planetoid, Actor, WikipediaNetwork, WebKB
import torch_geometric.transforms as T
# import os
# import re
#
# import networkx as nx
# import numpy as np
# import scipy.sparse as sp
# import torch as th
# from dgl import DGLGraph
# from sklearn.model_selection import ShuffleSplit
# from torch_geometric.data import Data
#
# from HPGT.utils.data_utils import preprocess_features

pl_name = ['Cora', 'CiteSeer', 'PubMed']
Wiki_name = ['chameleon', 'crocodile', 'squirrel']
Web_name = ['cornell', 'texas', 'wisconsin']


def get_dataset(name, Normalize=True):
    if name in pl_name:
        if Normalize == True:
            dataset = Planetoid(root='Data/'+name, name=name, transform=T.NormalizeFeatures())
        else:
            dataset = Planetoid(root='Data/'+name, name=name)
        return dataset
    if name == 'Actor':
        if Normalize == True:
            dataset = Actor(root='Data/Actor', transform=T.NormalizeFeatures())
        else:
            dataset = Actor(root='Data/Actor')
        return dataset
    if name in Wiki_name:
        if Normalize == True:
            dataset = WikipediaNetwork(root='Data/'+name, name=name, transform=T.NormalizeFeatures())
        else:
            dataset = WikipediaNetwork(root='Data/'+name, name=name)
        return dataset
    if name in Web_name:
        if Normalize==True:
            dataset = WebKB(root='Data/'+name, name=name,transform=T.NormalizeFeatures())
        else:
            dataset = WebKB(root='Data/'+name, name=name)
        return dataset
