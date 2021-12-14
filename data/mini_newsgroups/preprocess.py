# Preprocess mini_newsgroups
#
# Usage:
#   python preprocess.py
#

import collections
import copy
import glob
import logging
import os
import pickle
import random 

import cvxpy as cp
import higra as hg
import numba as nb
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from scipy.sparse import coo_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import AUROC

from IPython import embed


class SparseToDenseEncoder(pl.LightningModule):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = nn.Linear(in_dim, out_dim)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.auroc = AUROC(pos_label=1)

    def forward(self, x):
        embedding = self.encoder(x)
        return F.normalize(embedding)

    def _compute_all_pairs(self, batch):
        x, y = batch
        batch_size = x.shape[0]
        embeds = self(x)
        sims = embeds @ embeds.T
        lbls = (y[:, None] == y[None, :]).float()
        idx = torch.triu_indices(batch_size, batch_size, 1)
        return (sims[idx[0], idx[1]], lbls[idx[0], idx[1]])

    def training_step(self, batch, batch_idx):
        pred, target = self._compute_all_pairs(batch)
        loss = self.loss_fn(pred, target)
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred, target = self._compute_all_pairs(batch)
        val_loss = self.loss_fn(pred, target)
        val_auroc = self.auroc(pred, target.long())
        metrics = {'val_loss': val_loss, 'val_auroc': val_auroc}
        self.log_dict(metrics)
        return metrics

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_auroc = torch.stack([x['val_auroc'] for x in outputs]).mean()
        print('val_loss', avg_loss.item())
        print('val_auroc', avg_auroc.item())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
                self.parameters(),
                lr=1e-3,
                weight_decay=1e-4
        )
        return optimizer


@nb.njit(nogil=True, parallel=True)
def get_intra_cluster_energy(leaves, srcs, tgts, edge_weights, num_edges):
    total = 0.0
    for i in nb.prange(num_edges):
        if srcs[i] in leaves and tgts[i] in leaves:
            total += edge_weights[i]
    return total


def get_flat_clustering(tree, graph, edge_weights):
    n = graph.num_vertices()
    m = graph.num_edges()
    srcs, tgts = graph.edge_list()
    parents = tree.parents()
    membership = copy.deepcopy(parents[:n])
    best_clustering = np.arange(n)
    obj_vals = np.zeros((2*n - 1,))

    for node in tree.leaves_to_root_iterator(
            include_leaves=False, include_root=True):
        leaves_mask = (membership == node)
        leaves = np.where(leaves_mask)[0]
        obj_vals[node] = get_intra_cluster_energy(
                leaves, srcs, tgts, edge_weights, m)
        curr_obj_val = sum([obj_vals[i] 
            for i in np.unique(best_clustering[leaves_mask])])
        if obj_vals[node] > curr_obj_val:
            best_clustering[leaves_mask] = node
        membership[leaves_mask] = parents[node]

    return best_clustering


def get_max_agree_sdp_cc(graph, edge_weights):
    n = graph.num_vertices()

    logging.info('Constructing weight matrices')
    srcs, tgts = graph.edge_list()
    W = coo_matrix((edge_weights, (srcs, tgts)), shape=(n, n), dtype=np.double)
    
    # Define and solve the CVXPY specified optimization problem
    logging.info('Constructing optimization problem')
    X = cp.Variable((n, n), PSD=True)
    constraints = [
            cp.diag(X) == np.ones((n,)),
            X >= 0
    ]
    prob = cp.Problem(cp.Maximize(cp.trace(W @ X)), constraints)

    logging.info('Solving optimization problem')
    prob.solve(solver=cp.SCS, verbose=True)

    # run avg-linkage HAC on pairwise probabilities
    logging.info('Running HAC on pairwise probabilities')
    pp_graph, pp_edge_weights = hg.adjacency_matrix_2_undirected_graph(
            np.triu(X.value, k=1))
    tree, altitudes = hg.binary_partition_tree_average_linkage(
            pp_graph, 1.0-pp_edge_weights)

    # find best cut according to IC objective
    pred_clustering = get_flat_clustering(tree, graph, edge_weights)
    return pred_clustering


def shift_edge_weights(edge_weights, threshold):
    return edge_weights - threshold


def build_higra_complete_graph(embeds):
    # compute complete graph adjacency matrix
    adj_mx = coo_matrix(np.triu(embeds @ embeds.T, k=1))

    # build the graph
    graph = hg.UndirectedGraph(embeds.shape[0])
    graph.add_edges(adj_mx.row, adj_mx.col)
    edge_weights = adj_mx.data

    return (graph, edge_weights)


def get_best_threshold(embeds, labels):
    assert embeds.shape[0] == len(labels)
    graph, edge_weights = build_higra_complete_graph(embeds)
    thresholds = sorted(list(edge_weights))

    best_rand_idx = -1
    best_threshold = thresholds[0]

    while len(thresholds) > 1:
        quartile_idxs = [
                int(len(thresholds)/4),
                int(len(thresholds)/2), 
                int(3*len(thresholds)/4)
        ]
        quartile_thresholds = [thresholds[i] for i in quartile_idxs]
        quartile_rand_idxs = []
        for t in quartile_thresholds:
            tmp_edge_weights = shift_edge_weights(edge_weights, t)
            cand_clustering = get_max_agree_sdp_cc(graph, tmp_edge_weights)
            quartile_rand_idxs.append(
                    adjusted_rand_score(labels, cand_clustering)
            )
        best_quartile_idx = max(range(3), key=lambda i: quartile_rand_idxs[i])
        if quartile_rand_idxs[best_quartile_idx] > best_rand_idx:
            best_rand_idx = quartile_rand_idxs[best_quartile_idx]
            best_threshold = quartile_thresholds[best_quartile_idx]

        # shrink the thresholds list
        if best_quartile_idx == 0:
            thresholds = thresholds[:quartile_idxs[1]]
        elif best_quartile_idx == 1:
            thresholds = thresholds[quartile_idxs[0]+1:quartile_idxs[2]]
        elif best_quartile_idx == 2:
            thresholds = thresholds[quartile_idxs[1]+1:]
        else:
            raise ValueError('Something went wrong with quartile_idxs')

        print('Best Rand Idx: ', best_rand_idx)
        print('Best Threshold: ', best_threshold)
        print('Len Thresholds: ', len(thresholds))
        
        embed()
        exit()

    return best_threshold


def get_doc_text(doc_path):
    with open(doc_path, 'r', encoding='utf8', errors='ignore') as f:
        full_doc = f.read()
        header_end = full_doc.find('\n\n')
        body = full_doc[header_end:].strip()
    return body


def sparse_matrix_to_sparse_tensor(matrix):
    mx = matrix.tocoo()
    values = mx.data
    indices = np.vstack((mx.row, mx.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = mx.shape
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


if __name__ == '__main__':
    seed = 42
    pl.utilities.seed.seed_everything(seed)

    train_size = 50
    dev_size = 25
    test_size = 25

    files_by_split = collections.defaultdict(list)
    labels_by_split = collections.defaultdict(list)

    for group in glob.glob('raw/mini_newsgroups/*'):
        group_name = group.split('/')[-1]
        group_files = glob.glob(group + '/*')

        assert len(group_files) == train_size + dev_size + test_size
        random.shuffle(group_files)
        
        files_by_split['train'].extend(group_files[:train_size])
        files_by_split['dev'].extend(
                group_files[train_size:train_size+dev_size])
        files_by_split['test'].extend(group_files[-test_size:])

        labels_by_split['train'].extend([group_name] * train_size)
        labels_by_split['dev'].extend([group_name] * dev_size)
        labels_by_split['test'].extend([group_name] * test_size)

    train_docs = [get_doc_text(doc_path) 
            for doc_path in files_by_split['train']]
    dev_docs = [get_doc_text(doc_path) 
            for doc_path in files_by_split['dev']]
    test_docs = [get_doc_text(doc_path) 
            for doc_path in files_by_split['test']]

    corpus = train_docs + dev_docs + test_docs
    vectorizer = TfidfVectorizer(stop_words=['english'], max_df=250, min_df=3)
    vectorizer.fit_transform(corpus)
    
    train_sparse_embeds = vectorizer.transform(train_docs)
    dev_sparse_embeds = vectorizer.transform(dev_docs)
    test_sparse_embeds = vectorizer.transform(test_docs)

    train_feats_tensor = sparse_matrix_to_sparse_tensor(train_sparse_embeds)
    dev_feats_tensor = sparse_matrix_to_sparse_tensor(dev_sparse_embeds)
    test_feats_tensor = sparse_matrix_to_sparse_tensor(test_sparse_embeds)

    # remap split label names to ids
    all_labels = set([label for label in labels_by_split['test']])
    label_id_map = {v: i for i, v in enumerate(all_labels)}
    train_labels = [label_id_map[label] for label in labels_by_split['train']]
    dev_labels = [label_id_map[label] for label in labels_by_split['dev']]
    test_labels = [label_id_map[label] for label in labels_by_split['test']]

    # get dataset stuff organized for training
    train_ds = TensorDataset(train_feats_tensor, torch.LongTensor(train_labels))
    dev_ds = TensorDataset(dev_feats_tensor, torch.LongTensor(dev_labels))

    train_loader = DataLoader(train_ds, batch_size=100, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=500)

    ## train the model
    #dense_dim = 64
    #model = SparseToDenseEncoder(train_sparse_embeds.shape[1], dense_dim)
    #model.train()
    #checkpoint_callback = ModelCheckpoint(
    #        monitor='val_auroc',
    #        dirpath='model_checkpoints/',
    #        filename='sparse_to_dense_encoder-{epoch:04d}-{val_auroc:.4f}',
    #        save_top_k=1,
    #        mode='max'
    #)
    #trainer = pl.Trainer(
    #        max_epochs=100,
    #        check_val_every_n_epoch=1,
    #        callbacks=[checkpoint_callback]
    #)
    #trainer.fit(model, train_loader, dev_loader)
    #best_model_path = checkpoint_callback.best_model_path
    
    # load the best model
    best_model_path = '/mnt/nfs/work1/brun/rangell/ecc/data/mini_newsgroups/model_checkpoints/sparse_to_dense_encoder-epoch=0030-val_auroc=0.9099.ckpt'
    print('Best model path: ', best_model_path)
    model = SparseToDenseEncoder.load_from_checkpoint(best_model_path)
    model.eval()

    # get all dense dev embeddings
    with torch.no_grad():
        dev_dense_embeds = model(dev_feats_tensor).detach().numpy()

    # find the best threshold on dev
    threshold = get_best_threshold(dev_dense_embeds, dev_labels)

    print('Best threshold: ', threshold)

    embed()
    exit()

