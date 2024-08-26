#%%
from __future__ import division
from __future__ import print_function
import os
import argparse
import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
from utils import accuracy,sparse_mx_to_torch_sparse_tensor
from models.GCN import GCN
from models.ARGNN import ARGNN
from dataset import Dataset, get_PtbAdj, get_Ptbfeature

### package for AGE
import os, sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
SEED = 15
import argparse
import time
import datetime
import random
np.random.seed(SEED)
torch.manual_seed(SEED)

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
        default=False, help='debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=12, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--estimator', type=str, default='MLP',
                    choices=['MLP','GCN'])
parser.add_argument('--mlp_hidden', type=int, default=64,
                    help='Number of hidden units of MLP graph constructor')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default="cora",
                    choices=['cora', 'cora_ml', 'citeseer', 'acm'], help='dataset')
parser.add_argument('--attack_structure', type=str, default='random',
                    choices=['meta', 'random', 'nettack', "no"])
parser.add_argument('--attack_feature',type=str,default='no',choices=['no', 'meta', 'nettack'])
parser.add_argument("--label_rate", type=float, default=0.01, 
                    help='rate of labeled data')
parser.add_argument('--ptb_rate', type=float, default=0.15, 
                    help="noise ptb_rate")
parser.add_argument("--val_rate", type=float, default=0.1, help="size of validation set")
parser.add_argument("--test_rate", type=float, default=0.8, help="size of testing set")
parser.add_argument('--epochs', type=int,  default=1000, 
                    help='Number of epochs to train.')

parser.add_argument('--alpha', type=float, default=0.01, 
                    help='weight of rec loss')
parser.add_argument('--sigma', type=float, default=100, 
                    help='the parameter to control the variance of sample weights in rec loss')
parser.add_argument('--beta', type=float, default=0.3, help='weight of label smoothness loss')
parser.add_argument("--eta", type=float, default=1, help="the parameter to control the graoh encoder loss")
parser.add_argument('--threshold', type=float, default=0.6, 
                    help='threshold for adj of label smoothing')
parser.add_argument('--t_small',type=float, default=0.1,
                    help='threshold of eliminating the edges')

parser.add_argument('--inner_steps', type=int, default=2, 
                    help='steps for inner optimization')
parser.add_argument('--outer_steps', type=int, default=1, 
                    help='steps for outer optimization')
parser.add_argument('--lr_adj', type=float, default=0.001, 
                    help='lr for training adj')
parser.add_argument("--n_p", type=int, default=100, 
                    help='number of positive pairs per node')
parser.add_argument("--n_n", type=int, default=50, 
                    help='number of negitive pairs per node')

parser.add_argument("--r_type",type=str,default="flip",
                    choices=['add','remove','flip'])
### parameter for AGE
parser.add_argument('--gnnlayers', type=int, default=8, help="Number of gnn layers")
parser.add_argument('--linlayers', type=int, default=1, help="Number of hidden layers")
#parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')
parser.add_argument('--dims', type=int, default=[500], help='Number of units in hidden layer 1.')
#parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--upth_st', type=float, default=0.0110, help='Upper Threshold start.')
parser.add_argument('--lowth_st', type=float, default=0.1, help='Lower Threshold start.')
parser.add_argument('--upth_ed', type=float, default=0.001, help='Upper Threshold end.')
parser.add_argument('--lowth_ed', type=float, default=0.5, help='Lower Threshold end.')
parser.add_argument('--upd', type=int, default=10, help='Update epoch.')
parser.add_argument('--bs', type=int, default=10000, help='Batchsize.')
args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

if args.cuda:
    torch.cuda.manual_seed(args.seed)
if args.ptb_rate == 0:
    args.attack_structure = "no"
    args.attack_feature = "no"
print(args)

np.random.seed(15) # Here the random seed is to split the train/val/test data, we need to set the random seed to be the same as that when you generate the perturbed graph

data = Dataset(root='./data', name=args.dataset, val_rate=args.val_rate, test_rate=args.test_rate)
data_name = args.dataset
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
print(f"length idx train, idx val, idx test: {len(idx_train)}, {len(idx_val)}, {len(idx_test)}")

# change the selection for select labeled nodes
idx_train = idx_train[:int(args.label_rate * adj.shape[0])]
print(f"length labeled idx train: {len(idx_train)}")

if args.attack_structure == 'no' and args.attack_feature == "no":
    perturbed_adj = adj
    perturbed_feature = features

if args.attack_structure == 'random':
    ptb_rate = 0.3
    # from deeprobust.graph.global_attack import Random
    # import random
    # random.seed(15)
    # attacker = Random()
    # n_perturbations = int(args.ptb_rate * (adj.sum()//2))
    # attacker.attack(adj, n_perturbations, type=args.r_type)
    # perturbed_adj = attacker.modified_adj
    # perturbed_feature = features
    # file_path = "./data/{}_{}/{}/{}_{}_adj_{}.npz".format(args.val_rate * 100, args.test_rate * 100,
    #                                                       args.label_rate, args.dataset, args.attack, ptb_rate)
    # sp.save_npz(file_path,perturbed_adj.tocsr())
    perturbed_adj = get_PtbAdj(root="./data/{}_{}/{}".format(args.val_rate * 100, args.test_rate * 100, args.label_rate),
                               name=args.dataset,
                               attack_method=args.attack_structure,
                               ptb_rate=ptb_rate)
    perturbed_feature = features

if args.attack_structure in ['meta','nettack'] and args.attack_feature == "no":
    perturbed_adj = get_PtbAdj(root="./data/{}_{}/{}".format(args.val_rate * 100, args.test_rate * 100, args.label_rate),
            name=args.dataset,
            attack_method=args.attack_structure,
            ptb_rate=args.ptb_rate)
    perturbed_feature = features
    print(f"ptb adj == clean adj: {np.array(perturbed_adj.data) == np.array(adj.data)}")
    print(f"ptb feature == clean feature:{(np.array(perturbed_feature.data)==np.array(features.data))}")
    
if args.attack_feature in ["meta", "nettack"] and args.attack_structure == "no":
    perturbed_feature = get_Ptbfeature(root="./data/{}_{}/{}".format(args.val_rate * 100, args.test_rate * 100, args.label_rate),
                      name=args.dataset,
                      attack_method=args.attack_feature,
                      ptb_rate=args.ptb_rate)
    perturbed_adj = adj
    print(f"ptb adj == clean adj: {np.array(perturbed_adj.data) == np.array(adj.data)}")
    print(f"ptb feature == clean feature:{(np.array(perturbed_feature.data)==np.array(features.data))}")

if args.attack_structure in ["meta", "nettack"] and args.attack_feature in ["meta", "nettack"]:
    perturbed_adj = get_PtbAdj(root="./data/{}_{}/{}".format(args.val_rate * 100, args.test_rate * 100, args.label_rate),
            name=args.dataset,
            attack_method=args.attack_structure,
            ptb_rate=args.ptb_rate)
    
    perturbed_feature = get_Ptbfeature(root="./data/{}_{}/{}".format(args.val_rate * 100, args.test_rate * 100, args.label_rate),
                      name=args.dataset,
                      attack_method=args.attack_feature,
                      ptb_rate=args.ptb_rate)
    print(f"ptb adj == clean adj: {np.array(perturbed_adj.data) == np.array(adj.data)}")
    print(f"ptb feature == clean feature:{(np.array(perturbed_feature.data)==np.array(features.data))}")
#%%
import random
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
model = GCN(nfeat=args.dims[0],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            self_loop=True,
            dropout=args.dropout, device=device).to(device)

argnn = ARGNN(model,args,device)
argnn.fit(perturbed_feature, perturbed_adj, labels, idx_train, idx_val)
print("=====Test set accuracy=======")
acc_test = argnn.test(idx_test)

'''if want to save the result, uncomment the following code'''
# all_record_path = "./result.csv"
# if not os.path.exists(all_record_path):
#     all_record = pd.DataFrame({"seed": args.seed, "dataset": args.dataset, "attack_type": args.attack_structure, 
#                                "ptb_rate": args.ptb_rate, "attack_feature": args.attack_feature, "model": "ARGNN", "val_size": args.val_rate,
#                                "test_size": args.test_rate, "accuracy": round(acc_test, 3), 
#                                "laplayers": args.gnnlayers, "K": args.n_p, "upth_st": args.upth_st, "upth_ed": args.upth_ed, "lowth_st": args.lowth_st, "lowth_ed": args.lowth_ed,
#                                "Time": datetime.datetime.now(), "all_args": args}, index=[0])
#     all_record.to_csv(all_record_path, index=False)
# else:
#     all_record = pd.read_csv(all_record_path)
#     record = pd.DataFrame({"seed": args.seed, "dataset": args.dataset, "attack_type": args.attack_structure, 
#                                "ptb_rate": args.ptb_rate, "attack_feature": args.attack_feature, "model": "ARGNN", "val_size": args.val_rate,
#                                "test_size": args.test_rate, "accuracy": round(acc_test, 3),
#                            "laplayers": args.gnnlayers, "K": args.n_p, "upth_st": args.upth_st, "upth_ed": args.upth_ed, "lowth_st": args.lowth_st, "lowth_ed": args.lowth_ed,
#                            "Time": datetime.datetime.now(), "all_args": args}, index=[0])
#     all_record = pd.concat([all_record, record], axis=0, ignore_index=True)
#     all_record.to_csv(all_record_path, index=False)
# print("Result has been save to result.csv")


# %%