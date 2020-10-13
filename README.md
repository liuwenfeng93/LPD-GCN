# Locality Preserving Dense Graph Convolutional Networks with Graph Context-Aware Node Representations

This repository is the official PyTorch implementation of the experiments in the following paper: 

Wenfeng Liu, Maoguo Gong, Zedong Tang, A. K. Qin. Locality Preserving Dense Graph Convolutional Networks with Graph Context-Aware Node Representations.

[arXiv](https://arxiv.org/abs/2010.05404)

This work has been submitted to Neural Networks(https://www.journals.elsevier.com/neural-networks/). If you make use of the code/experiment or LPD-GCN algorithm in your work, please cite our paper (Bibtex below).
```
@article{Liu2020Locality,
  title={Locality Preserving Dense Graph Convolutional Networks with Graph Context-Aware Node Representations},
  author={Wenfeng Liu, Maoguo Gong, Zedong Tang, A. K. Qin},
  journal={ArXiv},
  volume={abs/2010.05404},
  year={2020}
}
```

## Installation
Install PyTorch following the instuctions on the [official website] (https://pytorch.org/). The code has been tested over PyTorch 0.4.1 and 1.0.0 versions.

Then install the other dependencies.
```
pip install -r requirements.txt
```

## Test run
Run

```
python main.py
```

Default parameters are not the best performing-hyper-parameters. Hyper-parameters need to be specified through the commandline arguments. Please refer to our paper for the details of how we set the hyper-parameters.

Type

```
python main.py --help
```

to learn hyper-parameters to be specified.

