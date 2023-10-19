# Semi-supervised heterophilic graph representation learning model based on Graph Transformer (HPGT)
The work has been published in the journal of Computer Applications.

## Requirements

#### 1. Neural network libraries for GNNs

* [pytorch](https://pytorch.org/get-started/locally/)
* [pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

Please check your cuda version first and install the above libraries matching your cuda. If possible, we recommend to install the latest versions of these libraries.

## Data preparation

You can use download all datasets by using dataset classes in torch_geometric.datasets. 

For Cora, Citeseer and PubMed, please download them by using torch_geometric.datasets.Planetoid(root: str, name: str, split: str = 'public', num_train_per_class: int = 20, num_val: int = 500, num_test: int = 1000, transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None).

For Cornell, Texas and Wisconsin, please download them by using torch_geometric.datasets.WebKB(root: str, name: str, transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None)

For Actor, please download it by using torch_geometric.datasets.Actor(root: str, transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None).
