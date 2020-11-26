# Locality Preserving Dense Graph Convolutional Networks with Graph Context-Aware Node Representations

This is the repo for the PyTorch implementation of the algorithm in the following paper: 

Wenfeng Liu, Maoguo Gong, Zedong Tang, A. K. Qin. Locality Preserving Dense Graph Convolutional Networks with Graph Context-Aware Node Representations.

## Abstract
Graph convolutional networks (GCNs) have been widely used for representation learning on graph data, which can capture structural patterns on a graph via specifically designed convolution and readout operations. In many graph classification applications, GCN-based approaches have outperformed traditional methods. However, most of the existing GCNs are inefficient to preserve local information of graphs -- a limitation that is especially problematic for graph classification. In this work, we propose a locality-preserving dense GCN with graph context-aware node representations. Specifically, our proposed model incorporates a local node feature reconstruction module to preserve initial node features into node representations, which is realized via a simple but effective encoder-decoder mechanism. To capture local structural patterns in neighbourhoods representing different ranges of locality, dense connectivity is introduced to connect each convolutional layer and its corresponding readout with all previous convolutional layers. To enhance node representativeness, the output of each convolutional layer is concatenated with the output of the previous layer's readout to form a global context-aware node representation. In addition, a self-attention module is introduced to aggregate layer-wise representations to form the final representation. Experiments on benchmark datasets demonstrate the superiority of the proposed model over state-of-the-art methods in terms of classification accuracy.

For more theoretical and pratical details about this repo, please go to [arXiv] (https://arxiv.org/abs/2010.05404), thanks!

## Installation
Install PyTorch following the instuctions on the [official website] (https://pytorch.org/). The code has been tested over PyTorch 0.4.1 and 1.0.0 versions.

Then install the other dependencies.
```
pip install -r requirements.txt
```

## Test run
Unzip the datasets file
```
unzip datasets.zip
```

and run

```
python main.py
```

Default parameters are not the best performing-hyper-parameters. Hyper-parameters need to be specified through the commandline arguments. Please refer to our paper for the details of how we set the hyper-parameters.

Type

```
python main.py --help
```

to learn hyper-parameters to be specified.

## Acknowledgement
This repository is modified from GIN(https://github.com/weihua916/powerful-gnns). We sincerely thank them for their contributions.

## Cite
This work has been submitted to Neural Networks (https://www.journals.elsevier.com/neural-networks/). If you make use of the code/experiment or LPD-GCN algorithm in your work, please cite our paper (Bibtex below).
```
@article{Liu2020Locality,
  title={Locality Preserving Dense Graph Convolutional Networks with Graph Context-Aware Node Representations},
  author={Wenfeng Liu, Maoguo Gong, Zedong Tang, A. K. Qin},
  journal={ArXiv},
  volume={abs/2010.05404},
  year={2020}
}
```
