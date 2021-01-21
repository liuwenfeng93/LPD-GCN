# If you have any question, please send email to 415525133@qq.com
# -*- coding: utf-8 -*-
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import time
from tqdm import tqdm

from util import load_data, separate_data
from models.gcn import GraphCNN

criterion = nn.CrossEntropyLoss()

def train(args, model, device, train_graphs, optimizer, epoch):
    model.train()

    total_iters = args.iters_per_epoch
    pbar = tqdm(range(total_iters), unit='batch')

    #batch_idx = np.random.permutation(len(train_graphs))

    h_loss_accum = 0
    loss_accum = 0
    for pos in pbar:
        #selected_idx = batch_idx[pos*args.batch_size:(pos+1)*args.batch_size]
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]

        batch_graph = [train_graphs[idx] for idx in selected_idx]
        [g_score, h_score] = model(batch_graph)

        g_labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)
        h_labels = []
        for graph in batch_graph:
            h_labels.extend(graph.node_tags)
        h_labels = torch.LongTensor(h_labels).to(device)

        # compute loss, which is the weighted sum of h_loss and g_loss controlled by Lambda.
        Lambda = 0.2
        g_loss = criterion(g_score, g_labels)
        h_loss = criterion(h_score, h_labels) #h_labels are node labels of all nodes over all graphs.
        
        loss = g_loss + Lambda*h_loss
        
        #backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()         
            optimizer.step()
        
        h_loss = h_loss.detach().cpu().numpy()
        h_loss_accum += h_loss

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

        #report
        pbar.set_description('epoch: %d' % (epoch))


    average_h_loss = h_loss_accum/total_iters

    average_loss = loss_accum/total_iters
    print("Total loss training: %f  h_loss: %f" % (average_loss,average_h_loss))
    
    return average_loss, average_h_loss

###pass data to model with minibatch during testing to avoid memory overflow (does not perform backpropagation)
def pass_data_iteratively(model, graphs, minibatch_size = 64):
    model.eval()
    g_output = []
    h_output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        g_output.append(model([graphs[j] for j in sampled_idx])[0].detach())
        h_output.append(model([graphs[j] for j in sampled_idx])[1].detach())
    return torch.cat(g_output, 0), torch.cat(h_output, 0)

def test(args, model, device, train_graphs, test_graphs, epoch):
    model.eval()
    # train accuracy
    [g_output, h_output] = pass_data_iteratively(model, train_graphs)

    g_pred = g_output.max(1, keepdim=True)[1]
    g_labels = torch.LongTensor([graph.label for graph in train_graphs]).to(device)
    g_correct = g_pred.eq(g_labels.view_as(g_pred)).sum().cpu().item()
    g_acc_train = g_correct / float(len(train_graphs))

    h_pred = h_output.max(1, keepdim=True)[1]
    h_labels = []
    for graph in train_graphs:
        h_labels.extend(graph.node_tags)
    h_labels = torch.LongTensor(h_labels).to(device)
    #h_labels = torch.LongTensor(h_labels.append(graph.node_tags) for graph in train_graphs).to(device)
    h_correct = h_pred.eq(h_labels.view_as(h_pred)).sum().cpu().item()
    h_acc_train = h_correct / float(len(h_labels))

    # test accuracy
    [g_output, h_output] = pass_data_iteratively(model, test_graphs)

    g_pred = g_output.max(1, keepdim=True)[1]
    g_labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    g_correct = g_pred.eq(g_labels.view_as(g_pred)).sum().cpu().item()
    g_acc_test = g_correct / float(len(test_graphs))

    h_pred = h_output.max(1, keepdim=True)[1]
    h_labels = []
    for graph in test_graphs:
        h_labels.extend(graph.node_tags)
    h_labels = torch.LongTensor(h_labels).to(device)
    h_correct = h_pred.eq(h_labels.view_as(h_pred)).sum().cpu().item()
    h_acc_test = h_correct / float(len(h_labels))

    print("graph classification accuracy train: %f test: %f" % (g_acc_train, g_acc_test))
    print("node classification accuracy train: %f test: %f" % (h_acc_train, h_acc_test))

    return g_acc_train, g_acc_test, h_acc_train, h_acc_test

def main():
    # Training settings
    # Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
    parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--dataset', type=str, default="MUTAG",
                        help='name of dataset (default: MUTAG)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--iters_per_epoch', type=int, default=50,
                        help='number of iterations per each epoch (default: 50)')
    parser.add_argument('--epochs', type=int, default=350,
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='the index of fold in 10-fold validation. Should be less then 10.')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"],
                        help='Pooling for over nodes in a graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average", "max"],
                        help='Pooling for over neighboring nodes: sum, average or max')
    parser.add_argument('--learn_eps', action="store_true",
                                        help='Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')
    parser.add_argument('--degree_as_tag', action="store_true",
    					help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
    parser.add_argument('--filename', type = str, default = "",
                                        help='output file')
    args = parser.parse_args()

    #set up seeds and gpu device
    torch.manual_seed(0)
    np.random.seed(0)    
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    #graphs are the networkx collections of all graphs, num_classes is the number of graph classes.
    graphs, num_classes, num_node_label = load_data(args.dataset, args.degree_as_tag) 

    ##10-fold cross validation. Conduct an experiment on the fold specified by args.fold_idx.
    train_graphs, test_graphs = separate_data(graphs, args.seed, args.fold_idx)

    model = GraphCNN(args.num_layers, args.num_mlp_layers, train_graphs[0].node_features.shape[1], args.hidden_dim, num_classes, num_node_label, args.final_dropout, args.learn_eps, args.graph_pooling_type, args.neighbor_pooling_type, device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    if not args.filename == "":
        with open(args.filename, 'a') as f:
            f.write("epoch  time_per_epoch  train_loss  g_acc_train  g_acc_test  h_acc_train  h_acc_test")
            f.write("\n")
    for epoch in range(1, args.epochs + 1):
        scheduler.step()
        start = time.time()
        avg_loss, avg_h_loss = train(args, model, device, train_graphs, optimizer, epoch)
        end = time.time()
        time_per_epoch = end - start
        [g_acc_train, g_acc_test, h_acc_train, h_acc_test] = test(args, model, device, train_graphs, test_graphs, epoch)

        if not args.filename == "":
            with open(args.filename, 'a') as f:
                f.write("%d  %f  %f  %f  %f  %f  %f" % (epoch, time_per_epoch, avg_loss, g_acc_train, g_acc_test, h_acc_train, h_acc_test))
                f.write("\n")
        print("")

        #epsilons = model.eps
        with open("h_loss_"+str(args.fold_idx)+".csv", 'a') as ff:
            ff.write("%f" % (avg_h_loss))
            ff.write("\n")
        print(model.eps)
    torch.save(model, 'models/model.pkl')
    

if __name__ == '__main__':
    main()
