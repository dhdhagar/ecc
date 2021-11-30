# Evaluate correlation clustering objective and optimization without
# there exists constraints comparing to HAC with tuned threshold.
#
# Usage:
#       python cc_control.py
#

import collections
import faiss
import higra as hg
import logging
import numba as nb
import numpy as np
import scipy
import sklearn
import time

from IPython import embed


def load_xcluster_tsv(infile, normalize=True):
    vecs = []
    lbls = []
    with open(infile, 'r') as f:
        for i, line in enumerate(f):
            splits = line.strip().split('\t')
            lbls.append(splits[1])
            vecs.append([float(x) for x in splits[2:]])
    vecs = np.array(vecs, dtype=np.float32)
    if normalize:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        num_zero = np.sum(norms == 0)
        logging.info('Loaded vectors, and unit norming. %s vectors had 0 norm.', num_zero)
        norms[norms == 0] = 1.0
        vecs /= norms
    return np.arange(vecs.shape[0]), vecs, lbls


def subsample_ds(ds, approx_size=2000):
    idxs, vecs, lbls = ds
    ds_size = idxs.size
    ss_percentage = approx_size / ds_size
    subsampled_ds, _ = split_tune_test(ds, tune_percentage=ss_percentage)
    return subsampled_ds


def split_tune_test(ds, tune_percentage=0.5):
    idxs, vecs, lbls = ds
    min_tune_size = int(tune_percentage * idxs.size)
    lbls2size = collections.Counter(lbls)

    lbls2size_keys = list(lbls2size.keys())
    total = 0
    for i, lbl in enumerate(lbls2size_keys):
        total += lbls2size[lbl]
        if total >= min_tune_size:
            break

    tune_lbl_set = set(lbls2size_keys[:i+1])
    test_lbl_set = set(lbls2size_keys[i+1:])

    tune_mask = np.asarray([lbl in tune_lbl_set for lbl in lbls])
    test_mask = np.asarray([lbl in test_lbl_set for lbl in lbls])

    tune_vecs = vecs[tune_mask]
    tune_idxs = np.arange(tune_vecs.shape[0])
    tune_lbls = [lbl for lbl in lbls if lbl in tune_lbl_set]

    test_vecs = vecs[test_mask]
    test_idxs = np.arange(tune_vecs.shape[0])
    test_lbls = [lbl for lbl in lbls if lbl in test_lbl_set]

    tune_ds = (tune_idxs, tune_vecs, tune_lbls)
    test_ds = (test_idxs, test_vecs, test_lbls)

    return tune_ds, test_ds


def build_knn_graph(ds, k):
    idxs, vecs, lbls = ds

    n, d = vecs.shape
    index = faiss.IndexFlatIP(d) 
    index.add(vecs)

    S, I = index.search(vecs, k+1)

    remove_mask = (idxs[:,None] == I)
    remove_mask[:,-1] = ~np.sum(idxs[:, None] == I, axis=1, dtype=bool)

    S = S[~remove_mask].reshape(n, k)
    I = I[~remove_mask].reshape(n, k)

    g = hg.UndirectedGraph(n)
    edge_weights = []
    for s, t_array, wt_array in zip(idxs, I, S):
        idx_gt_mask = t_array > s
        s_adj_tgts = t_array[idx_gt_mask]
        s_adj_srcs = np.full(s_adj_tgts.shape, s)
        g.add_edges(s_adj_srcs, s_adj_tgts)
        edge_weights.append(wt_array[idx_gt_mask])
    edge_weights = np.concatenate(edge_weights)

    # [-1, 1] -> [0, 1]
    edge_weights = (edge_weights / 2) + 0.5

    logging.info('Built kNN graph with %d nodes and %d edges',
                 g.num_vertices(), g.num_edges())

    return (g, edge_weights) 


def rescale_edge_weights(edge_weights, threshold):
    return np.clip(edge_weights - threshold + 0.5, 0, 1)


@nb.njit(nogil=True, parallel=True)
def _max_agree_objective(srcs, tgts, edge_weights, cand_clustering, num_edges):
    total = 0.0
    for i in nb.prange(num_edges):
        s = srcs[i]
        t = tgts[i]
        w = edge_weights[i]
        same_cluster = (cand_clustering[s] == cand_clustering[t])
        if same_cluster:
            total += (1 - w)
        else:
            total += w
    return total
        

def max_agree_objective(graph, edge_weights, cand_clustering):
    srcs, tgts = graph.edge_list()
    return _max_agree_objective(srcs, tgts, edge_weights,
                                cand_clustering, edge_weights.size)


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    dataset_fnames = [
        '/mnt/nfs/scratch1/nmonath/er/data/clustering/aloi.tsv.1',
        '/mnt/nfs/scratch1/nmonath/er/data/clustering/speaker_whitened.tsv.1',
        '/mnt/nfs/scratch1/nmonath/er/data/clustering/ilsvrc12_50k.tsv.1',
        '/mnt/nfs/scratch1/nmonath/er/data/clustering/covtype.tsv.1'
    ]

    ds = load_xcluster_tsv(dataset_fnames[1])
    tiny_ds = subsample_ds(ds, approx_size=2000)
    tune_ds, test_ds = split_tune_test(tiny_ds)

    k = 100
    logging.info('Building kNN graph on tune (k=%d)', k)
    tune_graph, tune_edge_weights = build_knn_graph(tune_ds, k)

    # Higra expects edge weights to be distances, so we subtract similarities
    # from 1.0 to get distances.
    logging.info('Building average linkage HAC tree...')
    tune_tree, tune_altitudes = hg.binary_partition_tree_average_linkage(
            tune_graph, 1.0-tune_edge_weights)

    # Cut the tree at an arbitrary threshold
    leaf_labels = hg.labelisation_horizontal_cut_from_threshold(
            tune_tree, tune_altitudes, 0.44)

    # Compute MaxAgree CC Objective Value
    objective_value = max_agree_objective(
            tune_graph, tune_edge_weights, leaf_labels)

    embed()
    exit()
