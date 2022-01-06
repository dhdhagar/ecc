# Run ecc simualtion using synthetically generated e-constraints.
#
# Usage:
#       python ecc.py
#

import copy
from itertools import product
import logging
import pickle
import time

import cvxpy as cp
import higra as hg
import numba as nb
from numba.typed import List
import numpy as np
import pytorch_lightning as pl
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
from scipy.sparse import vstack as sp_vstack
from scipy.special import softmax
from sklearn.metrics import adjusted_rand_score as rand_idx

from trellis import Trellis

from IPython import embed


class EccClusterer(object):

    def __init__(self, edge_weights: csr_matrix, features: np.ndarray):
        self.edge_weights = edge_weights.tocoo()
        self.features = features
        self.n = self.features.shape[0]
        self.ecc_constraints = []
        self.ecc_mx = None
        self.incompat_mx = None

    def add_constraint(self, ecc_constraint: csr_matrix):
        self.ecc_constraints.append(ecc_constraint)
        self.ecc_mx = sp_vstack(self.ecc_constraints)
        self.n += 1

    @staticmethod
    @nb.njit(parallel=True)
    def _set_incompat_mx(n: int,
                         m: int,
                         indptr_a: np.ndarray,
                         indices_a: np.ndarray,
                         data_a: np.ndarray,
                         indptr_b: np.ndarray,
                         indices_b: np.ndarray,
                         data_b: np.ndarray,
                         incompat_mx: np.ndarray):
        for i in nb.prange(n):
            for j in nb.prange(m):
                ptr_a = indptr_a[i]
                ptr_b = indptr_b[j]
                next_ptr_a = indptr_a[i+1]
                next_ptr_b = indptr_b[j+1]
                while ptr_a < next_ptr_a and ptr_b < next_ptr_b:
                    if indices_a[ptr_a] < indices_b[ptr_b]:
                        ptr_a += 1
                    elif indices_a[ptr_a] > indices_b[ptr_b]:
                        ptr_b += 1
                    else:
                        if data_a[ptr_a] * data_b[ptr_b] == -1:
                            incompat_mx[i, j] = True
                            break
                        ptr_a += 1
                        ptr_b += 1

    @staticmethod
    @nb.njit
    def _get_feat_satisfied_hyperplanes(feats_indptr: np.ndarray,
                                        feats_indices: np.ndarray,
                                        pos_ecc_indptr: np.ndarray,
                                        pos_ecc_indices: np.ndarray,
                                        incompat_mx: np.ndarray):
        ecc_indices = List.empty_list(nb.int64)
        points_indptr = List.empty_list(nb.int64)
        points_indices = List.empty_list(nb.int64)

        points_indptr.append(0)
        for ecc_idx in range(pos_ecc_indptr.size-1):
            pos_feat_ptr = pos_ecc_indptr[ecc_idx]
            next_pos_feat_ptr = pos_ecc_indptr[ecc_idx+1]
            while pos_feat_ptr < next_pos_feat_ptr:
                ecc_indices.append(ecc_idx)
                points_indptr.append(points_indptr[-1])
                feat_id = pos_ecc_indices[pos_feat_ptr]
                point_ptr = feats_indptr[feat_id]
                next_point_ptr = feats_indptr[feat_id+1]

                while point_ptr < next_point_ptr:
                    point_idx = feats_indices[point_ptr]
                    if not incompat_mx[point_idx, ecc_idx]:
                        points_indptr[-1] += 1
                        points_indices.append(point_idx)
                    point_ptr += 1
                pos_feat_ptr += 1

        return (ecc_indices, points_indptr, points_indices)

    def build_and_solve_sdp(self):
        num_points = self.features.shape[0]
        num_ecc = len(self.ecc_constraints)

        if num_ecc > 0:
            # "negative" sdp constraints
            uni_feats = sp_vstack([self.features, self.ecc_mx])
            self.incompat_mx = np.zeros(
                    (num_points+num_ecc, num_ecc), dtype=bool)
            self._set_incompat_mx(num_points+num_ecc,
                                  num_ecc,
                                  uni_feats.indptr,
                                  uni_feats.indices,
                                  uni_feats.data,
                                  self.ecc_mx.indptr,
                                  self.ecc_mx.indices,
                                  self.ecc_mx.data,
                                  self.incompat_mx)
            ortho_indices = [(a, b+num_points)
                    for a, b in zip(*np.where(self.incompat_mx))]

            # "positive" sdp constraints
            bin_features = self.features.astype(bool).tocsc()
            pos_ecc_mx = (self.ecc_mx > 0)
            (ecc_indices,
             points_indptr,
             points_indices) = self._get_feat_satisfied_hyperplanes(
                     bin_features.indptr,
                     bin_features.indices,
                     pos_ecc_mx.indptr,
                     pos_ecc_mx.indices,
                     self.incompat_mx)
            ecc_indices = [x + num_points for x in ecc_indices]
            points_indptr = list(points_indptr)
            points_indices = list(points_indices)

        # formulate SDP
        logging.info('Constructing optimization problem')
        W = csr_matrix((self.edge_weights.data,
                        (self.edge_weights.row, self.edge_weights.col)),
                        shape=(self.n, self.n))
        X = cp.Variable((self.n, self.n), PSD=True)
        # standard correlation clustering constraints
        constraints = [
                cp.diag(X) == np.ones((self.n,)),
                X >= 0
        ]
        if num_ecc > 0:
            # "negative" ecc constraints
            constraints.extend([X[i,j] <= 0 for i, j in ortho_indices])
            # "positive" ecc constraints
            for idx, i in enumerate(ecc_indices):
                j_s = points_indices[points_indptr[idx]: points_indptr[idx+1]]
                constraints.append(sum([X[i,j] for j in j_s]) >= 1)

        prob = cp.Problem(cp.Maximize(cp.trace(W @ X)), constraints)

        logging.info('Solving optimization problem')
        sdp_obj_value = prob.solve(solver=cp.SCS, verbose=True, max_iters=50000)

        pw_probs = X.value
        if self.incompat_mx is not None:
            # discourage incompatible nodes from clustering together
            self.incompat_mx = np.concatenate(
                    (np.zeros((num_points+num_ecc, num_points), dtype=bool),
                     self.incompat_mx), axis=1
            )
            pw_probs[self.incompat_mx] -= np.sum(pw_probs)
        pw_probs = np.triu(pw_probs, k=1)
        return sdp_obj_value, pw_probs

    def build_trellis(self, pw_probs: np.ndarray):
        num_trees = 5
        noise_lvl = 0.05
        t = Trellis(adj_mx=pw_probs, num_trees=num_trees, noise_lvl=noise_lvl)
        t.fit()
        return t

    def get_intra_cluster_energy(self, leaves: np.ndarray):
        row_mask = np.isin(self.edge_weights.row, leaves)
        col_mask = np.isin(self.edge_weights.col, leaves)
        data_mask = row_mask & col_mask
        return np.sum(self.edge_weights.data[data_mask])

    def get_num_ecc_sat(self, leaves: np.ndarray, num_points: int):
        point_leaves = leaves[leaves < num_points]
        ecc_indices = leaves[leaves >= num_points] - num_points
        if ecc_indices.squeeze().size == 0:
            return 0
        feats = get_cluster_feats(self.features[point_leaves])
        ecc_avail = self.ecc_mx[ecc_indices]
        to_satisfy = (ecc_avail > 0).sum(axis=1)
        num_ecc_sat = ((feats @ ecc_avail.T).T == to_satisfy).sum()
        return num_ecc_sat

    @staticmethod
    @nb.njit(parallel=True)
    def get_membership_data(indptr: np.ndarray,
                            indices: np.ndarray):
        data = np.empty(indices.shape, dtype=np.int64)
        for i in nb.prange(indptr.size-1):
            for j in nb.prange(indptr[i], indptr[i+1]):
                data[j] = i
        return data

    @staticmethod
    @nb.njit
    def merge_memberships(lchild_indices: np.ndarray,
                          lchild_data: np.ndarray,
                          rchild_indices: np.ndarray,
                          rchild_data: np.ndarray,
                          parent_indices: np.ndarray,
                          parent_data: np.ndarray):
        assert lchild_indices.size == lchild_data.size
        assert rchild_indices.size == rchild_data.size
        assert parent_indices.size == parent_data.size
        assert (lchild_data.size + rchild_data.size) == parent_data.size
        lchild_ptr = 0
        rchild_ptr = 0
        for i in range(parent_data.size):
            if (rchild_ptr == rchild_indices.size or
                    (lchild_ptr < lchild_indices.size and
                     lchild_indices[lchild_ptr] < rchild_indices[rchild_ptr])):
                assert parent_indices[i] == lchild_indices[lchild_ptr]
                parent_data[i] = lchild_data[lchild_ptr]
                lchild_ptr += 1
            else:
                assert parent_indices[i] == rchild_indices[rchild_ptr]
                assert (lchild_ptr == lchild_indices.size
                        or lchild_indices[lchild_ptr] != rchild_indices[rchild_ptr])
                parent_data[i] = rchild_data[rchild_ptr]
                rchild_ptr += 1

    def cut_trellis(self, t: Trellis):
        num_ecc = len(self.ecc_constraints)
        num_points = self.n - num_ecc

        membership_indptr = t.leaves_indptr
        membership_indices = t.leaves_indices
        membership_data = self.get_membership_data(membership_indptr,
                                                   membership_indices)
        obj_vals = np.zeros((t.num_nodes,))
        num_ecc_sat = np.zeros((t.num_nodes,))

        for node in t.internal_nodes_topo_ordered():
            node_start = membership_indptr[node]
            node_end = membership_indptr[node+1]
            leaves = membership_indices[node_start:node_end]
            if num_ecc > 0:
                num_ecc_sat[node] = self.get_num_ecc_sat(leaves, num_points)
            obj_vals[node] = self.get_intra_cluster_energy(leaves)
            for lchild, rchild in t.get_child_pairs_iter(node):
                cpair_num_ecc_sat = num_ecc_sat[lchild] + num_ecc_sat[rchild]
                cpair_obj_val = obj_vals[lchild] + obj_vals[rchild]
                if (num_ecc_sat[node] < cpair_num_ecc_sat 
                    or (num_ecc_sat[node] == cpair_num_ecc_sat
                        and obj_vals[node] < cpair_obj_val)):
                    num_ecc_sat[node] = cpair_num_ecc_sat
                    obj_vals[node] = cpair_obj_val
                    lchild_start = membership_indptr[lchild]
                    lchild_end = membership_indptr[lchild+1]
                    rchild_start = membership_indptr[rchild]
                    rchild_end = membership_indptr[rchild+1]
                    self.merge_memberships(
                            membership_indices[lchild_start:lchild_end],
                            membership_data[lchild_start:lchild_end],
                            membership_indices[rchild_start:rchild_end],
                            membership_data[rchild_start:rchild_end],
                            membership_indices[node_start:node_end],
                            membership_data[node_start:node_end],
                    )

        # The value of `node` is the root since we iterate over the trellis
        # nodes in topological order bottom up. Moreover, the values of
        # `node_start` and `node_end` also correspond to the root of the
        # trellis.
        best_clustering = membership_data[node_start:node_end]
        if num_ecc > 0:
            best_clustering = best_clustering[:-num_ecc]

        return best_clustering, obj_vals[node], num_ecc_sat[node]

    def pred(self):
        num_ecc = len(self.ecc_constraints)

        # Construct and solve SDP
        start_solve_time = time.time()
        sdp_obj_value, pw_probs = self.build_and_solve_sdp()
        end_solve_time = time.time()

        # Build trellis
        t = self.build_trellis(pw_probs)

        # Cut trellis
        pred_clustering, cut_obj_value, num_ecc_satisfied = self.cut_trellis(t)

        metrics = {
                'sdp_solve_time': end_solve_time - start_solve_time,
                'sdp_obj_value': sdp_obj_value,
                'cut_obj_value': cut_obj_value,
                'num_ecc_satisfied': int(num_ecc_satisfied),
                'num_ecc': num_ecc,
                'frac_ecc_satisfied': num_ecc_satisfied / num_ecc
                        if num_ecc > 0 else 0.0,
                'num_ecc_feats': self.ecc_mx.nnz 
                        if self.ecc_mx is not None else 0,
                'num_pos_ecc_feats': (self.ecc_mx > 0).nnz
                        if self.ecc_mx is not None else 0,
                'num_neg_ecc_feats': (self.ecc_mx < 0).nnz
                        if self.ecc_mx is not None else 0,
        }

        return pred_clustering, metrics


def get_cluster_feats(point_feats: csr_matrix):
    csc_indptr = point_feats.tocsc().indptr
    return csr_matrix((np.diff(csc_indptr) > 0).astype(np.int64))


@nb.njit(parallel=True)
def set_matching_matrix(gold_indptr: np.ndarray,
                        gold_indices: np.ndarray,
                        pred_indptr: np.ndarray,
                        pred_indices: np.ndarray,
                        matching_mx: np.ndarray) -> None:
    for i in nb.prange(gold_indptr.size - 1):
        for j in nb.prange(pred_indptr.size - 1):
            gold_ptr = gold_indptr[i]
            next_gold_ptr = gold_indptr[i+1]
            pred_ptr = pred_indptr[j]
            next_pred_ptr = pred_indptr[j+1]
            num_intersect = 0
            num_union = 0
            while gold_ptr < next_gold_ptr and pred_ptr < next_pred_ptr:
                if gold_indices[gold_ptr] < pred_indices[pred_ptr]:
                    gold_ptr += 1
                    num_union += 1
                elif gold_indices[gold_ptr] > pred_indices[pred_ptr]:
                    pred_ptr += 1
                    num_union += 1
                else:
                    gold_ptr += 1
                    pred_ptr += 1
                    num_intersect += 1
                    num_union += 1
            if gold_ptr < next_gold_ptr:
                num_union += (next_gold_ptr - gold_ptr)
            elif pred_ptr < next_pred_ptr:
                num_union += (next_pred_ptr - pred_ptr)
            matching_mx[i, j] = num_intersect / num_union
    return matching_mx


def gen_ecc_constraint(point_feats: csr_matrix,
                       gold_cluster_lbls: np.ndarray,
                       pred_cluster_lbls: np.ndarray,
                       gold_cluster_feats: csr_matrix,
                       pred_cluster_feats: csr_matrix,
                       matching_mx: np.ndarray,
                       max_overlap_feats: int,
                       max_pos_feats: int,
                       max_neg_feats: int,
                       overlap_col_wt: np.ndarray,
                       pos_col_wt: np.ndarray,
                       neg_col_wt: np.ndarray):

    # set perfect match rows and columns to zero so they will not be picked
    perfect_match = (matching_mx == 1.0)
    row_mask = np.any(perfect_match, axis=1)
    column_mask = np.any(perfect_match, axis=0)
    to_zero_mask = row_mask[:, None] | column_mask[None, :]
    matching_mx[to_zero_mask] = 0.0

    norm_factor = np.sum(matching_mx)
    if norm_factor == 0.0:
        logging.info('Features of gold clusters already fully satisfied.'
                     ' Cannot generate constraints to affect clustering.')
        logging.info('Exiting...')
        exit()

    # pick a gold cluster, pred cluster pair
    norm_matching_mx = matching_mx / norm_factor
    pair_ravel_idx = np.where(
            np.random.multinomial(1, norm_matching_mx.ravel()))
    gold_cluster_idx, pred_cluster_idx = np.unravel_index(
            pair_ravel_idx[0][0], matching_mx.shape)

    # select features to for the ecc constraint
    src_feats = pred_cluster_feats[pred_cluster_idx]
    tgt_feats = gold_cluster_feats[gold_cluster_idx]
    all_overlap_feats = ((src_feats + tgt_feats) == 2).astype(np.int64)
    all_pos_feats = ((tgt_feats - src_feats) == 1).astype(np.int64)
    all_neg_feats = ((src_feats - tgt_feats) == 1).astype(np.int64)

    def sample_csr_cols(mx, num_sample, col_wt=None):
        choices = mx.tocoo().col
        k = min(num_sample, choices.size)
        if k > num_sample:
            p = col_wt[choices]**1
            p /= np.sum(p)
        else:
            p = None
        samples = np.random.choice(choices, (k,), replace=False, p=p)
        return samples

    sampled_overlap_feats = sample_csr_cols(
            all_overlap_feats, max_overlap_feats, col_wt=overlap_col_wt)
    sampled_pos_feats = sample_csr_cols(
            all_pos_feats, max_pos_feats, col_wt=pos_col_wt)
    sampled_neg_feats = sample_csr_cols(
            all_neg_feats, max_neg_feats, col_wt=neg_col_wt)

    new_ecc_col = np.hstack(
            (sampled_overlap_feats,
             sampled_pos_feats,
             sampled_neg_feats)
    )
    new_ecc_row = np.zeros_like(new_ecc_col)
    new_ecc_data = np.hstack(
            (np.ones_like(sampled_overlap_feats),
             np.ones_like(sampled_pos_feats),
             -1*np.ones_like(sampled_neg_feats))
    )

    new_ecc = coo_matrix((new_ecc_data, (new_ecc_row, new_ecc_col)),
                         shape=src_feats.shape, dtype=np.int64).tocsr()

    # generate "equivalent" pairwise point constraints
    overlap_feats = set(sampled_overlap_feats)
    pos_feats = set(sampled_pos_feats)
    neg_feats = set(sampled_neg_feats)

    gold_cluster_points = set(np.where(gold_cluster_lbls==gold_cluster_idx)[0])
    pred_cluster_points = set(np.where(pred_cluster_lbls==pred_cluster_idx)[0])

    gold_and_pred = gold_cluster_points & pred_cluster_points
    gold_not_pred = gold_cluster_points - pred_cluster_points
    pred_not_gold = pred_cluster_points - gold_cluster_points

    num_points = point_feats.shape[0]
    pairwise_constraints = dok_matrix((num_points, num_points))
    for s, t in product(gold_and_pred, gold_not_pred):
        s_feats = set(point_feats[s].indices)
        t_feats = set(point_feats[t].indices)
        if not (s_feats.isdisjoint(overlap_feats) 
                or t_feats.isdisjoint(overlap_feats)
                or t_feats.isdisjoint(pos_feats)):
            if s < t:
                pairwise_constraints[s, t] = 1
            else:
                pairwise_constraints[t, s] = 1

    for s, t in product(gold_and_pred, pred_not_gold):
        s_feats = set(point_feats[s].indices)
        t_feats = set(point_feats[t].indices)
        if not (s_feats.isdisjoint(overlap_feats) 
                or t_feats.isdisjoint(overlap_feats)
                or t_feats.isdisjoint(neg_feats)):
            if s < t:
                pairwise_constraints[s, t] = -1
            else:
                pairwise_constraints[t, s] = -1

    return new_ecc, pairwise_constraints


def simulate(edge_weights: csr_matrix,
             point_features: csr_matrix,
             gold_clustering: np.ndarray):

    gold_cluster_feats = sp_vstack([
        get_cluster_feats(point_features[gold_clustering == i])
            for i in np.unique(gold_clustering)
    ])

    clusterer = EccClusterer(edge_weights=edge_weights,
                             features=point_features)

    # TODO: move these to hparams
    max_overlap_feats = 100
    max_pos_feats = 100
    max_neg_feats = 100

    ## create column weights
    # for now just do some uniform feature sampling
    feat_freq = np.array(point_features.sum(axis=0))
    overlap_col_wt = np.ones((point_features.shape[1],))
    pos_col_wt = np.ones((point_features.shape[1],))
    neg_col_wt = np.ones((point_features.shape[1],))
    #overlap_col_wt = feat_freq
    #pos_col_wt = feat_freq
    #neg_col_wt = feat_freq

    for r in range(100):
        # compute predicted clustering
        pred_clustering, metrics = clusterer.pred()

        # get predicted cluster feats and do some label remapping for later
        uniq_pred_cluster_lbls = np.unique(pred_clustering)
        pred_cluster_feats = sp_vstack([
            get_cluster_feats(point_features[pred_clustering == i])
                for i in uniq_pred_cluster_lbls
        ])
        remap_lbl_dict = {j: i for i, j in enumerate(uniq_pred_cluster_lbls)}
        for i in range(pred_clustering.size):
            pred_clustering[i] = remap_lbl_dict[pred_clustering[i]]

        # construct jaccard similarity matching matrix
        matching_mx = np.empty((gold_cluster_feats.shape[0],
                                pred_cluster_feats.shape[0]))
        set_matching_matrix(
                gold_cluster_feats.indptr, gold_cluster_feats.indices,
                pred_cluster_feats.indptr, pred_cluster_feats.indices,
                matching_mx
        )

        # handle some metric stuffs
        metrics['match_feat_coeff'] = np.mean(np.max(matching_mx, axis=1))
        metrics['rand_idx'] = rand_idx(gold_clustering, pred_clustering)
        metric_str = '; '.join([
            k + ' = ' 
            + ('{:.4f}'.format(v) if isinstance(v, float) else str(v))
                for k, v in metrics.items()
        ])
        logging.info('Round %d metrics - ' + metric_str, r)

        # exit if we predict the ground truth clustering
        if metrics['rand_idx'] == 1.0:
            assert metrics['match_feat_coeff'] == 1.0
            logging.info('Achieved perfect clustering at round %d.', r)
            break

        # generate a new constraint
        while True:
            ecc_constraint, pairwise_constraints = gen_ecc_constraint(
                    point_features,
                    gold_clustering,
                    pred_clustering,
                    gold_cluster_feats,
                    pred_cluster_feats,
                    matching_mx,
                    max_overlap_feats,
                    max_pos_feats,
                    max_neg_feats,
                    overlap_col_wt,
                    pos_col_wt,
                    neg_col_wt
            )
            already_exists = any([
                (ecc_constraint != x).nnz == 0
                for x in clusterer.ecc_constraints
            ])
            if not already_exists:
                break
            logging.warning('Produced duplicate ecc constraint')

        # add new constraint
        clusterer.add_constraint(ecc_constraint)


if __name__ == '__main__':
    logging.basicConfig(
            format='(ECC) :: %(asctime)s >> %(message)s',
            datefmt='%m-%d-%y %H:%M:%S',
            level=logging.INFO
    )

    seed = 42
    pl.utilities.seed.seed_everything(seed)

    data_fname = '../data/pubmed_processed.pkl'
    with open(data_fname, 'rb') as f:
        blocks_preprocessed = pickle.load(f)

    for block_name, block_data in blocks_preprocessed.items():
        edge_weights = block_data['edge_weights']
        point_features = block_data['point_features']
        gold_clustering = block_data['labels']

        assert edge_weights.shape[0] == point_features.shape[0]
        num_clusters = np.unique(gold_clustering).size

        logging.info(f'Loaded block \"{block_name}\"')
        logging.info(f'\t number of points: {edge_weights.shape[0]}')
        logging.info(f'\t number of clusters: {num_clusters}')
        logging.info(f'\t number of features: {point_features.shape[1]}')

        simulate(edge_weights, point_features, gold_clustering)


