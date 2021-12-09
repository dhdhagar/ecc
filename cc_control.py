# Evaluate correlation clustering objective and optimization without
# there exists constraints comparing to HAC with tuned threshold.
#
# Usage:
#       python cc_control.py
#

import copy
import collections
import faiss
import higra as hg
import logging
import time

import cvxpy as cp
import numba as nb
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.metrics import adjusted_rand_score

from IPython import embed


def load_xcluster_tsv(infile, normalize=True):
    vecs = []
    lbl_map = {}
    lbls = []
    with open(infile, 'r') as f:
        for i, line in enumerate(f):
            splits = line.strip().split('\t')
            lbl_key = splits[1]
            if lbl_key not in lbl_map.keys():
                lbl_map[lbl_key] = len(lbl_map)
            lbls.append(lbl_map[lbl_key])
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


@nb.njit(nogil=True, parallel=True)
def rewire_graph(adj_list, rewire_mask, n, m):
    for i in nb.prange(n):
        for j in nb.prange(m):
            if rewire_mask[i][j]:
                tgt = adj_list[i][j]
                edges = adj_list[i]
                while True:
                    tgt = np.random.randint(m)
                    contained = False
                    for e in edges:
                        if tgt == e:
                            contained = True
                            break
                    if not contained:
                        break
                adj_list[i][j] = tgt
    return adj_list


@nb.njit(nogil=True, parallel=True)
def compute_similarity_matrix(adj_list, vecs, n, m):
    sims_mx = np.empty(adj_list.shape, dtype=np.double)
    for i in nb.prange(n):
        for j in nb.prange(m):
            sims_mx[i][j] = np.dot(vecs[i], vecs[j])
    return sims_mx


def build_knn_sw_graph(ds, k, p):
    idxs, vecs, lbls = ds

    n, d = vecs.shape
    index = faiss.IndexFlatIP(d) 
    index.add(vecs)

    _, adj_list = index.search(vecs, k+1)

    # rewire w/ probability `p`
    rewire_mask = (np.random.rand(*adj_list.shape) < p)
    adj_list = rewire_graph(adj_list, rewire_mask, n, k+1)

    # compute `sims_mx` matrix given rewired `adj_list`
    sims_mx = compute_similarity_matrix(adj_list, vecs, n, k+1)

    # remove self loops
    remove_mask = (idxs[:,None] == adj_list)
    remove_mask[:,-1] = ~np.sum(remove_mask[:,:-1], axis=1, dtype=bool)
    sims_mx = sims_mx[~remove_mask].reshape(n, k)
    adj_list = adj_list[~remove_mask].reshape(n, k)

    # build higra graph
    g = hg.UndirectedGraph(n)
    edge_weights = []
    for s, t_array, wt_array in zip(idxs, adj_list, sims_mx):
        idx_gt_mask = t_array > s
        s_adj_tgts = t_array[idx_gt_mask]
        s_adj_srcs = np.full(s_adj_tgts.shape, s)
        g.add_edges(s_adj_srcs, s_adj_tgts)
        edge_weights.append(wt_array[idx_gt_mask])
    edge_weights = np.concatenate(edge_weights)

    # [-1, 1] -> [0, 1]
    edge_weights = (edge_weights / 2) + 0.5

    logging.info('Built kNN+SW graph with %d nodes and %d edges',
                 g.num_vertices(), g.num_edges())

    return (g, edge_weights) 


def shift_edge_weights(edge_weights, threshold):
    return edge_weights - threshold


def max_agree_objective(graph, edge_weights, cand_clustering):
    srcs, tgts = graph.edge_list()
    return _max_agree_objective(srcs, tgts, edge_weights,
                                cand_clustering, edge_weights.size)


@nb.njit(nogil=True, parallel=True)
def _max_agree_objective(srcs, tgts, edge_weights, cand_clustering, num_edges):
    total = 0.0
    for i in nb.prange(num_edges):
        if cand_clustering[srcs[i]] == cand_clustering[tgts[i]]:
            total += edge_weights[i]
    return total


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

    embed()
    exit()


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



if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    dataset_fnames = [
        '/mnt/nfs/scratch1/nmonath/er/data/clustering/aloi.tsv.1',
        '/mnt/nfs/scratch1/nmonath/er/data/clustering/speaker_whitened.tsv.1',
        '/mnt/nfs/scratch1/nmonath/er/data/clustering/ilsvrc12_50k.tsv.1',
        '/mnt/nfs/scratch1/nmonath/er/data/clustering/covtype.tsv.1'
    ]

    ds = load_xcluster_tsv(dataset_fnames[1])
    tiny_ds = subsample_ds(ds, approx_size=200)
    tune_ds, test_ds = split_tune_test(tiny_ds)

    k = 20
    p = 0.5
    threshold = 0.5

    logging.info('Building kNN+SW graph on tune (k=%d, p=%f)', k, p)
    tune_graph, tune_edge_weights = build_knn_sw_graph(tune_ds, k, p)

    # shift edge weights using given threshold
    tune_edge_weights = shift_edge_weights(tune_edge_weights, threshold)

    cand_clustering = get_max_agree_sdp_cc(tune_graph, tune_edge_weights)

    logging.info('Done.')
    exit()


    # Higra expects edge weights to be distances, so we subtract similarities
    # from 1.0 to get distances.
    logging.info('Building average linkage HAC tree...')
    tune_tree, tune_altitudes = hg.binary_partition_tree_average_linkage(
            tune_graph, 1.0-tune_edge_weights)

    # Cut the tree at an arbitrary threshold
    for threshold in [0.39, 0.4, 0.41, 0.42, 0.43, 0.44]:
        cand_clustering = hg.labelisation_horizontal_cut_from_threshold(
                tune_tree, tune_altitudes, threshold)

        # Compute MaxAgree CC Objective Value
        objective_value = max_agree_objective(
                tune_graph, tune_edge_weights, cand_clustering)

        # Compute Adjusted Rand Index
        _, _, gold_labels = tune_ds
        adj_rand_idx = adjusted_rand_score(gold_labels, cand_clustering)

        logging.info('Threshold: %f - Objective Value: %f - '
                     'Adjusted Rand Index: %f', threshold, objective_value,
                     adj_rand_idx)

    embed()
    exit()
