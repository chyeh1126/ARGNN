#%%
import torch
import numpy as np
import torch.nn.functional as F
import pandas as pd
import datetime
import torch.optim as optim
from deeprobust.graph.defense import GCN
from deeprobust.graph.targeted_attack import Nettack
from deeprobust.graph.utils import *
# from deeprobust.graph.data import Dataset
from dataset import Dataset
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'cora_ml', 'citeseer', "acm"], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.15,  help='pertubation rate')
parser.add_argument("--label_rate", type=float, default=0.01, help='rate of labeled data')
parser.add_argument("--val_rate", type=float, default=0.1, help="size of validation set")
parser.add_argument("--test_rate", type=float, default=0.8, help="size of testing set")


args = parser.parse_known_args()[0]
args.cuda = torch.cuda.is_available()
print('cuda: %s' % args.cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(args)

np.random.seed(15)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)



data = Dataset(root='./data/', name=args.dataset, val_rate=args.val_rate, test_rate=args.test_rate)
adj, features, labels = data.adj, data.features, data.labels

if args.dataset in ["texas","cornell"]:
    args.label_rate = 0.05
    
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
print("length of train index: {}".format(len(idx_train)))
idx_train = idx_train[:int(args.label_rate * adj.shape[0])]
print("length of labeled train index: {}".format(len(idx_train)))
print("length of val index: {}".format(len(idx_val)))
print("length of test index: {}".format(len(idx_test)))
# Setup Surrogate model
surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                nhid=16, dropout=0, with_relu=False, with_bias=False, device=device)

surrogate = surrogate.to(device)
surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)

def test(adj, features, target_node):
    ''' test on GCN '''
    gcn = GCN(nfeat=features.shape[1],
              nhid=16,
              nclass=labels.max().item() + 1,
              dropout=0.5, device=device)

    gcn = gcn.to(device)

    gcn.fit(features, adj, labels, idx_train, idx_val, patience=10)

    gcn.eval()
    output = gcn.predict()
    probs = torch.exp(output[[target_node]])[0]
    print('Target node probs: {}'.format(probs.detach().cpu().numpy()))
    acc_test = accuracy(output[idx_test], labels[idx_test])

    print("Overall test set results:",
          "accuracy= {:.4f}".format(acc_test.item()))

    return acc_test.item()

def multi_test_poison():
    # test on 40 nodes on poisoining attack
    cnt = 0
    degrees = adj.sum(0).A1
    np.random.seed(42)
    idx = np.arange(0,adj.shape[0])
    np.random.shuffle(idx)
    node_list = idx[:int(args.ptb_rate*len(idx))]

    num = len(node_list)
    print('=== [Poisoning] Attacking %s nodes respectively ===' % num)

    modified_adj = adj
    for target_node in tqdm(node_list):
        n_perturbations = int(degrees[target_node])
        model = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=False, attack_features=True, device=device)
        model = model.to(device)

        model.attack(features, modified_adj, labels, target_node, n_perturbations, verbose=False)
        modified_adj = model.modified_adj
        modified_features = model.modified_features
        acc = single_test(modified_adj, modified_features, target_node)
        if acc == 0:
            cnt += 1
    print('misclassification rate : %s' % (cnt/num))
    import os
    import scipy.sparse as sp
    path = os.path.join("./data/{}".format(args.label_rate),"nettack/")
    if not os.path.exists(path):
        os.makedirs(path)
    file_path = os.path.join(path,"{}.npz".format(args.dataset))
    # if type(modified_adj) is torch.Tensor:
    #     sparse_adj = to_scipy(modified_adj)
    #     sp.save_npz(file_path, sparse_adj)
    # else:
    #     sp.save_npz(file_path, modified_adj)



def single_test(adj, features, target_node, gcn=None):
    if gcn is None:
        # test on GCN (poisoning attack)
        gcn = GCN(nfeat=features.shape[1],
                  nhid=16,
                  nclass=labels.max().item() + 1,
                  dropout=0.5, device=device)

        gcn = gcn.to(device)

        gcn.fit(features, adj, labels, idx_train, idx_val, patience=30)
        gcn.eval()
        output = gcn.predict()
    else:
        # test on GCN (evasion attack)
        output = gcn.predict(features, adj)
    probs = torch.exp(output[[target_node]])

    # acc_test = accuracy(output[[target_node]], labels[target_node])
    acc_test = (output.argmax(1)[target_node] == labels[target_node])
    return acc_test.item()

#%%
cnt = 0
degrees = adj.sum(0).A1
np.random.seed(42)
idx = np.arange(0,adj.shape[0])
np.random.shuffle(idx)
node_list = idx[:int(args.ptb_rate*len(idx))]
# node_list=[0]

modified_adj = adj
modified_features = features
num = len(node_list)
print('=== [Poisoning] Attacking %s nodes respectively ===' % num)
for target_node in tqdm(node_list):
    # print("degree of node {}: {}".format(target_node, degrees[target_node]))
    n_perturbations = int(degrees[target_node])
    if n_perturbations == 0:
        continue
    model = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=True, device=device)
    model = model.to(device)
    model.attack(modified_features, modified_adj, labels, target_node, n_perturbations, verbose=False)
    modified_adj = model.modified_adj
    modified_features = model.modified_features
    modified_features = modified_features.tocsr()
    
    acc = single_test(modified_adj, modified_features, target_node)
    if acc == 0:
        cnt += 1

print("Attack structure")
print(modified_adj.data == adj.data)
print("if modified adj is different from original adj: {}".format((np.array(modified_adj.data)==np.array(adj.data))))
print("the same location: {}".format(np.where(np.array(modified_adj.data)==np.array(adj.data))))

print("Attack feature")
print((modified_features.data == features.data))
print("if modified adj is different from original adj: {}".format((np.array(modified_features.data)==np.array(features.data))))
print("the same location: {}".format(np.where(np.array(modified_features.data)==np.array(features.data))))

print('misclassification rate : %s' % (cnt/num))

#%%
import os
import scipy.sparse as sp
path = os.path.join("./data/{}_{}/{}".format(args.val_rate * 100, args.test_rate * 100, args.label_rate))
if not os.path.exists(path):
    os.makedirs(path)
adj_path = os.path.join(path,"{}_nettack_adj_{}.npz".format(args.dataset,args.ptb_rate))
feature_path = os.path.join(path,"{}_nettack_feature_{}.npz".format(args.dataset,args.ptb_rate))
sp.save_npz(adj_path, modified_adj.tocsr())
sp.save_npz(feature_path,modified_features.tocsr())