#%%
import os
import time
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import *
import torch_geometric.utils as utils
from models.GCN import GCN
import scipy.sparse as sp
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
#### package for age ####
from optimizer import loss_function
from lintrans import LinTrans
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn import metrics
from tqdm import tqdm

class ARGNN:
    def __init__(self, model, args, device):
        self.device = device
        self.args = args
        self.best_val_acc = 0
        self.best_val_loss = 10
        self.best_graph = None
        self.best_hidden_emb = None
        self.hidden_emb = None
        self.weights = None
        self.estimator = None
        self.model = model.to(device)
        self.model_emb = None

    def fit(self, features, adj, labels, idx_train, idx_val): #features=hidden_emb
        """Train RS-GNN.

        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices
        """
        args = self.args
        edge_index, _ = utils.from_scipy_sparse_matrix(adj)
        edge_index = edge_index.to(self.device)

        if sp.issparse(features):
            features = sparse_mx_to_torch_sparse_tensor(features).to_dense().float()
        else:
            features = torch.FloatTensor(np.array(features))
        features = features.to(self.device)
        labels = torch.LongTensor(np.array(labels)).to(self.device)

        self.features = features
        self.labels = labels
        self.edge_index = edge_index
        self.adj = adj
        
        ##### Laplacian #####
        print("Laplacian Smoothing...")
        n_nodes, feat_dim = features.shape
        adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
        adj.eliminate_zeros()

        n = adj.shape[0]
        adj_norm_s = preprocess_graph(adj, args.gnnlayers, norm='sym', renorm=True) #generate filter matrix
        sm_fea_s = sp.csr_matrix(features.cpu()).toarray()

        for a in adj_norm_s:
            sm_fea_s = a.dot(sm_fea_s)  #smoothed feature matrix
        adj_1st = (adj + sp.eye(n)).toarray()
        adj_label = torch.FloatTensor(adj_1st)

        sm_fea_s = torch.FloatTensor(sm_fea_s)
        adj_label = adj_label.reshape([-1,])

        if args.cuda:
            #model.cuda()
            inx = sm_fea_s.to(self.device)
            adj_label = adj_label.to(self.device)

        pos_num = len(adj.indices)
        neg_num = n_nodes*n_nodes-pos_num

        

        up_eta = (args.upth_ed - args.upth_st) / (args.epochs/args.upd)
        low_eta = (args.lowth_ed - args.lowth_st) / (args.epochs/args.upd)

        self.pos_inds, self.neg_inds = self.update_similarity(normalize(sm_fea_s.numpy()), args.upth_st, args.lowth_st, pos_num, neg_num)
        self.upth, self.lowth = self.update_threshold(args.upth_st, args.lowth_st, up_eta,low_eta)
        
        self.estimator = EstimateAdj(edge_index, features, args, device=self.device).to(self.device) #use origin features matrix to get perturbed edge
        self.model_emb = LinTrans(layers=args.linlayers,dims=[features.shape[1]]+args.dims)
        self.model_emb = self.model_emb.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(),
                               lr=args.lr, weight_decay=args.weight_decay)
        self.optimizer_adj = optim.Adam(list(self.estimator.parameters()) + list(self.model_emb.parameters()),
                               lr=args.lr_adj, weight_decay=args.weight_decay)


        bs = min(args.bs, len(self.pos_inds))
        length = len(self.pos_inds)
    
        self.pos_inds_cuda = torch.LongTensor(self.pos_inds).to(self.device)
        # Train model
        t_total = time.time()
        with tqdm(total=args.epochs, desc="Training") as pbar:
            for epoch in range(args.epochs):
                st, ed = 0, bs
                batch_num = 0
                self.model_emb.train()
                length = len(self.pos_inds)
                for i in range(int(args.outer_steps)):
                    self.train_adj(epoch, adj, sm_fea_s, edge_index, labels,idx_train, idx_val,
                                   length,n_nodes,st,ed,bs,batch_num,inx,up_eta,low_eta,pos_num,neg_num,
                                   self.neg_inds,self.upth,self.lowth,self.pos_inds_cuda)

                for i in range(int(args.inner_steps)):
                    self.train_gcn(epoch, features, edge_index,
                            labels, idx_train, idx_val)

                pbar.update(1)

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        # Testing
        print("picking the best model according to validation performance")
        self.model.load_state_dict(self.weights)

        print("=====validation set accuracy=======")
        self.test(idx_val)
        print("===================================")

    def train_gcn(self, epoch, features, edge_index, labels, idx_train, idx_val):
        args = self.args

        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(self.hidden_emb, self.estimator.poten_edge_index, self.estimator.estimated_weights.detach())
        acc_train = accuracy(output[idx_train], labels[idx_train])


        loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
        ''' for ablation study, remember to uncomment '''
        loss_label_smooth = self.label_smoothing(self.estimator.poten_edge_index,\
                                                 self.estimator.estimated_weights.detach(),\
                                                 output, idx_train, self.args.threshold)
        loss = loss_train  + self.args.beta * loss_label_smooth
        loss.backward()
        self.optimizer.step()

        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        self.model.eval()
        output = self.model(self.hidden_emb, self.estimator.poten_edge_index, self.estimator.estimated_weights.detach())

        loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        

        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_graph = self.estimator.estimated_weights.detach()
            self.best_hidden_emb = self.hidden_emb
            self.weights = deepcopy(self.model.state_dict())
            if args.debug:
                print('\t=== saving current graph/gcn, best_val_acc: %s' % self.best_val_acc.item())

        if args.debug:
            if epoch % 1 == 0:
                print('Epoch: {:04d}'.format(epoch+1),
                      'loss_train: {:.4f}'.format(loss_train.item()),
                      'acc_train: {:.4f}'.format(acc_train.item()),
                      'loss_val: {:.4f}'.format(loss_val.item()),
                      'acc_val: {:.4f}'.format(acc_val.item()),
                      'time: {:.4f}s'.format(time.time() - t))



    def train_adj(self, epoch, adj, features, edge_index, labels, idx_train, idx_val, length,n_nodes,st,ed,bs,batch_num,inx,up_eta,low_eta,pos_num,neg_num,neg_inds,upth,lowth,pos_inds_cuda):
        args = self.args
        if args.debug:
            print("\n=== train_adj ===")
        t = time.time()

        features = features.to(self.device)
        labels = self.labels
        n_nodes, feat_dim = features.shape

        up_eta = up_eta
        low_eta = low_eta

        ed = ed
        length = length
        if ed > length:
            ed = length
        
        while ( ed <= length ):
            sampled_neg = torch.LongTensor(np.random.choice(neg_inds, size=ed-st)).to(self.device)
            sampled_inds = torch.cat((pos_inds_cuda[st:ed], sampled_neg), 0) 
            t = time.time()
            xind = sampled_inds // n_nodes
            yind = sampled_inds % n_nodes
            x = torch.index_select(inx, 0, xind)
            y = torch.index_select(inx, 0, yind)
            zx = self.model_emb(x)
            zy = self.model_emb(y)
            batch_label = torch.cat((torch.ones(ed-st), torch.zeros(ed-st))).cuda()
            batch_pred = self.model_emb.dcs(zx, zy)

            loss_encoder = loss_function(adj_preds=batch_pred, adj_labels=batch_label, n_nodes=ed-st)

            st = ed
            batch_num += 1

            if ed < length and ed + bs >= length:
                ed += length - ed
            else:
                ed += bs

        if (epoch + 1) % args.upd == 0: #evaluate every 10 times
            self.model_emb.eval()
            mu = self.model_emb(inx) #size:3327*500
            hidden_emb = mu.cpu().data.numpy() #size:3327*500,the feature matrix which want to pass to next step
        #filter
            self.upth, self.lowth = self.update_threshold(self.upth, self.lowth, up_eta, low_eta)
            self.pos_inds, self.neg_inds = self.update_similarity(hidden_emb, self.upth, self.lowth, pos_num, neg_num) #function above
            self.bs = min(args.bs, len(self.pos_inds)) #10000
            self.pos_inds_cuda = torch.LongTensor(self.pos_inds).to(self.device) #variable
        else:
            hidden_emb = self.model_emb(features) #this is smoothed feature matrix
        self.hidden_emb = torch.tensor(hidden_emb,dtype=torch.float32).to(self.device)
        


        self.optimizer_adj.zero_grad()
        
        rec_loss = self.estimator(edge_index,self.hidden_emb)
        output = self.model(self.hidden_emb, self.estimator.poten_edge_index, self.estimator.estimated_weights)
        loss_gcn = F.cross_entropy(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])

        loss_label_smooth = self.label_smoothing(self.estimator.poten_edge_index,\
                                                 self.estimator.estimated_weights.detach(),\
                                                 output, idx_train, self.args.threshold)
        ''' add regularizer to encoder loss'''
        total_loss = loss_gcn + args.alpha * rec_loss + loss_encoder * args.eta


        total_loss.backward()

        self.optimizer_adj.step()


        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        self.model.eval()
        output = self.model(self.hidden_emb, self.estimator.poten_edge_index, self.estimator.estimated_weights.detach())

        loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        

        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_graph = self.estimator.estimated_weights.detach()
            self.best_hidden_emb = self.hidden_emb
            self.weights = deepcopy(self.model.state_dict())
            if args.debug:
                print('\t=== saving current graph/gcn, best_val_acc: %s' % self.best_val_acc.item())


        if args.debug:
            if epoch % 1 == 0:
                print('Epoch: {:04d}'.format(epoch+1),
                      'loss_gcn: {:.4f}'.format(loss_gcn.item()),
                      'rec_loss: {:.4f}'.format(rec_loss.item()),
                      'loss_label_smooth: {:.4f}'.format(loss_label_smooth.item()),
                      'loss_total: {:.4f}'.format(total_loss.item()))
                print('Epoch: {:04d}'.format(epoch+1),
                        'acc_train: {:.4f}'.format(acc_train.item()),
                        'loss_val: {:.4f}'.format(loss_val.item()),
                        'acc_val: {:.4f}'.format(acc_val.item()),
                        'time: {:.4f}s'.format(time.time() - t))

    def update_similarity(self, z, upper_threshold, lower_threshold, pos_num, neg_num):
        f_adj = np.matmul(z, np.transpose(z)) #3327*3327 feature matrix
        cosine = f_adj
        cosine = cosine.reshape([-1,]) #change

        pos_num = round(upper_threshold * len(cosine))
        neg_num = round((1-lower_threshold) * len(cosine))

        pos_inds = np.argpartition(-cosine, pos_num)[:pos_num]
        neg_inds = np.argpartition(cosine, neg_num)[:neg_num]
    
        return np.array(pos_inds), np.array(neg_inds)

    def update_threshold(self, upper_threshold, lower_threshold, up_eta, low_eta):
        upth = upper_threshold + up_eta
        lowth = lower_threshold + low_eta
        return upth, lowth

    def test(self, idx_test):
        """Evaluate the performance of ProGNN on test set
        """
        print("\t=== testing ===")
        args = self.args
        features = self.features
        labels = self.labels
        edge_index = self.edge_index
        self.model.eval()
        self.model_emb.eval()
        estimated_weights = self.best_graph
        if self.best_graph is None:
            estimated_weights = self.estimator.estimated_weights

        #### link predictor ####

        #estimated_weights_array = estimated_weights.cpu().numpy()
        #list_poten_edge_index = np.array(list(zip(self.estimator.poten_edge_index.cpu().numpy()[0],self.estimator.poten_edge_index.cpu().numpy()[1])))
        #list_edge_index = np.array(list(zip(edge_index.cpu().numpy()[0],edge_index.cpu().numpy()[1])))
        #result = [i for i, brow in enumerate(list_poten_edge_index) for srow in list_edge_index if all(srow == brow)]
        #raw_weight = np.zeros(estimated_weights.size())
        #raw_weight[result] = 1
        #link_predictor_mse = mean_squared_error(estimated_weights_array,raw_weight)
        #print("link predictor mse: ",link_predictor_mse)

        #output_feature = self.model_emb()
        #output_emb = self.model_emb(features)
        output = self.model(self.best_hidden_emb, self.estimator.poten_edge_index, estimated_weights)
        loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print("Results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return float(acc_test)

    
    def label_smoothing(self, edge_index, edge_weight, representations, idx_train, threshold):


        num_nodes = representations.shape[0]
        n_mask = torch.ones(num_nodes, dtype=torch.bool).to(self.device)
        n_mask[idx_train] = 0

        mask = n_mask[edge_index[0]] \
                & (edge_index[0] < edge_index[1])\
                & (edge_weight >= threshold)\
                | torch.bitwise_not(n_mask)[edge_index[1]]

        unlabeled_edge = edge_index[:,mask]
        unlabeled_weight = edge_weight[mask]

        Y = F.softmax(representations)

        loss_smooth_label = unlabeled_weight\
                            @ torch.pow(Y[unlabeled_edge[0]] - Y[unlabeled_edge[1]], 2).sum(dim=1)\
                            / num_nodes

        return loss_smooth_label
                        
#%%
class EstimateAdj(nn.Module):
    """Provide a pytorch parameter matrix for estimated
    adjacency matrix and corresponding operations.
    """

    def __init__(self, edge_index, features, args ,device='cuda'): #use original feature matrix
        super(EstimateAdj, self).__init__()

        
        if args.estimator=='MLP':
            self.estimator = nn.Sequential(nn.Linear(args.dims[0],args.mlp_hidden), #change feature.shape[1] to 500
                                    nn.ReLU(),
                                    nn.Linear(args.mlp_hidden,args.mlp_hidden))
        else:
            self.estimator = GCN(500, args.mlp_hidden, args.mlp_hidden,dropout=0.0,device=device)
        self.device = device
        self.args = args
        self.poten_edge_index = self.get_poten_edge(edge_index,features,args.n_p)
        self.features_diff = None#torch.cdist(features,features,2)
        self.estimated_weights = None


    def get_poten_edge(self, edge_index, features, n_p):

        if n_p == 0:
            return edge_index

        poten_edges = []
        for i in range(len(features)):
            sim = torch.div(torch.matmul(features[i],features.T), features[i].norm()*features.norm(dim=1))
            _,indices = sim.topk(n_p)
            poten_edges.append([i,i])
            indices = set(indices.cpu().numpy())
            indices.update(edge_index[1,edge_index[0]==i])
            for j in indices:
                if j > i:
                    pair = [i,j]
                    poten_edges.append(pair)
        poten_edges = torch.as_tensor(poten_edges).T

        poten_edges = utils.to_undirected(poten_edges, len(features)).to(self.device)

        return poten_edges
    

    def forward(self, edge_index, features): #use learned embedding matrix
        if self.args.estimator=='MLP':
            representations = self.estimator(features)
        else:
            representations = self.estimator(features,edge_index,\
                                            torch.ones([edge_index.shape[1]]).to(self.device).float())
        self.features_diff = torch.cdist(features,features,2)
        rec_loss = self.reconstruct_loss(edge_index, representations)

        x0 = representations[self.poten_edge_index[0]]
        x1 = representations[self.poten_edge_index[1]]
        output = torch.sum(torch.mul(x0,x1),dim=1)

        self.estimated_weights = F.relu(output)
        self.estimated_weights[self.estimated_weights < self.args.t_small].data = torch.Tensor([0.0])
        

        return rec_loss
    
    def reconstruct_loss(self, edge_index, representations):
        
        num_nodes = representations.shape[0]
        randn = utils.negative_sampling(edge_index,num_nodes=num_nodes, num_neg_samples=self.args.n_n*num_nodes)
        randn = randn[:,randn[0]<randn[1]]

        edge_index = edge_index[:, edge_index[0]<edge_index[1]]
        neg0 = representations[randn[0]]
        neg1 = representations[randn[1]]
        neg = torch.sum(torch.mul(neg0,neg1),dim=1)

        pos0 = representations[edge_index[0]]
        pos1 = representations[edge_index[1]]
        pos = torch.sum(torch.mul(pos0,pos1),dim=1)

        neg_loss = torch.exp(torch.pow(self.features_diff[randn[0],randn[1]]/self.args.sigma,2)) @ F.mse_loss(neg,torch.zeros_like(neg), reduction='none')
        pos_loss = torch.exp(-torch.pow(self.features_diff[edge_index[0],edge_index[1]]/self.args.sigma,2)) @ F.mse_loss(pos, torch.ones_like(pos), reduction='none')

        rec_loss = (pos_loss + neg_loss) \
                    * num_nodes/(randn.shape[1] + edge_index.shape[1]) 
        

        return rec_loss


# %%