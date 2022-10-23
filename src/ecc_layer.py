# Run ecc simualtion using synthetically generated e-constraints as a cvxpylayer.
#
# Usage:
#       python ecc_layer.py --data_path ../data/arnetminer_processed.pkl \
#           --output_dir="../experiments/exp_ecc_arnetminer" --max_sdp_iters=50000 \
#           --max_rounds=1 --only_avg_hac
#

import argparse
from collections import defaultdict
import copy
import heapq
from itertools import product
import json
import logging
import os
import pickle
import time

import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer
import higra as hg
import numba as nb
from numba.typed import List
import numpy as np
import pytorch_lightning as pl
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
from scipy.sparse import vstack as sp_vstack
from scipy.special import softmax
from sklearn.metrics import adjusted_rand_score as rand_idx
from sklearn.metrics import homogeneity_completeness_v_measure as cluster_f1

from trellis import Trellis

from IPython import embed


class TrellisCutLayer(torch.nn.Module):
    """
    Takes the SDP solution as input and executes the trellis-cut rounding algorithm in the forward pass
    Executes a straight-through estimator in the backward pass
    """
    def __init__(self, ecc_clusterer_obj):
        super().__init__()
        self.clusterer = ecc_clusterer_obj

    def get_rounded_solution(self, pw_probs, only_avg_hac=False):
        t = self.clusterer.build_trellis(pw_probs.detach().numpy(), only_avg_hac=only_avg_hac)
        pred_clustering, cut_obj_value, num_ecc_satisfied = self.clusterer.cut_trellis(t)

        self.cut_obj_value = cut_obj_value
        self.num_ecc_satisfied = num_ecc_satisfied
        self.pred_clustering = pred_clustering

        # Return an NxN matrix of 0s and 1s based on the predicted clustering
        round_mat = torch.eye(len(self.pred_clustering), dtype=torch.float)
        _v = None
        _i = None
        for i, v in enumerate(self.pred_clustering):
            if v == _v:
                round_mat[_i, i] = 1.
                round_mat[i, _i] = 1.
            else:
                _i = i
                _v = v
        return round_mat

    def forward(self, X, only_avg_hac=False):
        return X + (self.get_rounded_solution(X, only_avg_hac) - X).detach()

class EccClusterer(object):

    def __init__(self,
                 edge_weights: csr_matrix,
                 features: np.ndarray,
                 max_num_ecc: int,
                 max_pos_feats: int,
                 max_sdp_iters: int):

        self.edge_weights = edge_weights.tocoo()
        self.features = features
        self.max_num_ecc = max_num_ecc
        self.max_pos_feats = max_pos_feats
        self.max_sdp_iters = max_sdp_iters
        self.num_points = self.features.shape[0]
        self.num_ecc = 0
        self.ecc_constraints = []

        self.ecc_mx = None
        self.incompat_mx = None

        n = self.num_points + self.max_num_ecc

        # formulate SDP
        logging.info('Constructing optimization problem')

        W = csr_matrix((self.edge_weights.data, (self.edge_weights.row, self.edge_weights.col)), shape=(n, n))
        self.W_val = torch.tensor(W.todense(), requires_grad=True)

        self.X = cp.Variable((n, n), PSD=True)

        self.W = cp.Parameter((n, n))

        # # instantiate parameters
        # self.L = cp.Parameter((self.max_num_ecc, n))  # lower
        # self.U = cp.Parameter((self.max_num_ecc, n))  # upper
        # self.As = [cp.Parameter((n, self.max_num_ecc))
        #         for _ in range(self.max_pos_feats)]
        #
        # # initialize parameters
        # self.L.value = np.zeros((self.max_num_ecc, n))
        # self.U.value = np.ones((self.max_num_ecc, n))
        # for A in self.As:
        #     A.value = np.ones((n, self.max_num_ecc))

        # build out constraint set
        constraints = [
                cp.diag(self.X) == np.ones((n,)),
                self.X[:self.num_points, :] >= 0,
                # self.X[self.num_points:, :] >= self.L,
                # self.X[self.num_points:, :] <= self.U,
        ]
        # constraints.extend([
        #     cp.diag(self.X[self.num_points:, :] @ A)
        #         >= np.ones((self.max_num_ecc,)) for A in self.As
        # ])

        # create problem
        self.prob = cp.Problem(cp.Maximize(cp.trace(self.W @ self.X)), constraints)
        # Note: maximizing the trace is equivalent to maximizing the sum_E (w_uv * X_uv) objective
        # because W is upper-triangular and X is symmetric

        # Build the SDP cvxpylayer
        self.sdp_layer = CvxpyLayer(self.prob, parameters=[self.W], variables=[self.X])
        # Build the trellis-cut rounding layer
        self.rounding_layer = TrellisCutLayer(ecc_clusterer_obj=self)

    def add_constraint(self, ecc_constraint: csr_matrix):
        self.ecc_constraints.append(ecc_constraint)
        self.ecc_mx = sp_vstack(self.ecc_constraints)
        self.num_ecc += 1

        # "negative" sdp constraints
        uni_feats = sp_vstack([self.features, self.ecc_mx])
        self.incompat_mx = np.zeros(
                (self.num_points+self.num_ecc, self.num_ecc), dtype=bool)
        self._set_incompat_mx(self.num_points+self.num_ecc,
                              self.num_ecc,
                              uni_feats.indptr,
                              uni_feats.indices,
                              uni_feats.data,
                              self.ecc_mx.indptr,
                              self.ecc_mx.indices,
                              self.ecc_mx.data,
                              self.incompat_mx)

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

        # if there if no way a single cluster can satisfy both constraints...
        for idx, i in enumerate(ecc_indices):
            pos_feats_points = points_indices[
                    points_indptr[idx]:
                    points_indptr[idx+1]
            ]
            for j in range(i+1, self.num_ecc):
                all_pos_incompat = True
                for l in pos_feats_points:
                    if not self.incompat_mx[l, j]:
                        all_pos_incompat = False
                        break
                if all_pos_incompat:
                    self.incompat_mx[i, j] = True
                    self.incompat_mx[j, i] = True

        # update problem constraint parameters
        active_U = self.U.value[:self.num_ecc, :self.num_points+self.num_ecc]
        active_U[self.incompat_mx.T] = 0.0

        self.var_vals = defaultdict(list)
        pos_feat_idx = 0
        curr_ecc_idx = 0
        for ptr_idx, i in enumerate(ecc_indices):
            if i != curr_ecc_idx:
                pos_feat_idx = 0
                curr_ecc_idx = i
            start, end = points_indptr[ptr_idx], points_indptr[ptr_idx+1]
            self.As[pos_feat_idx].value[:, i] = 0.0
            for j in points_indices[start:end]:
                self.As[pos_feat_idx].value[j, i] = 1.0
                self.var_vals[(pos_feat_idx, i)].append(j)
            pos_feat_idx += 1

        self.var_vals = dict(
                sorted(self.var_vals.items(), key=lambda x: len(x[1]))
        )

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
        logging.info('Solving optimization problem')

        # Forward pass through the SDP cvxpylayer
        pw_probs = self.sdp_layer(self.W_val, solver_args={
            "solve_method": "SCS",
            "verbose": True,
            # "warm_start": True,  # Enabled by default
            "max_iters": self.max_sdp_iters,
            "eps": 1e-3
        })
        pw_probs = torch.triu(pw_probs[0], diagonal=1)
        with torch.no_grad():
            sdp_obj_value = torch.sum(self.W_val * pw_probs).item()

        # sdp_obj_value = self.prob.solve(
        #         solver=cp.SCS,
        #         verbose=True,
        #         warm_start=True,
        #         max_iters=self.max_sdp_iters,
        #         eps=1e-3
        # )

        # number of active graph nodes we are clustering
        active_n = self.num_points + self.num_ecc

        # run heuristic max forcing for now
        if self.num_ecc > 0:
            var_assign = []
            for ((_, ecc_idx), satisfying_points) in self.var_vals.items():
                max_satisfy_pt = max(
                        satisfying_points,
                        key=lambda x: self.X.value[ecc_idx+self.num_points, x]
                )
                var_assign.append((ecc_idx, max_satisfy_pt))

            for ecc_idx, point_idx in var_assign:
                self.L.value[ecc_idx, point_idx] = 0.9

            misdp_obj_value = self.prob.solve(
                    solver=cp.SCS,
                    verbose=True,
                    warm_start=True,
                    max_iters=self.max_sdp_iters,
                    eps=1e-3
            )
            pw_probs = self.X.value[:active_n, :active_n]

            for ecc_idx, point_idx in var_assign:
                self.L.value[ecc_idx, point_idx] = 0.0
        else: 
            # pw_probs = self.X.value[:active_n, :active_n]
            pw_probs = pw_probs[:active_n, :active_n]

        if self.incompat_mx is not None:
            # discourage incompatible nodes from clustering together
            self.incompat_mx = np.concatenate(
                    (np.zeros((active_n, self.num_points), dtype=bool),
                     self.incompat_mx), axis=1
            )
            pw_probs[self.incompat_mx] -= np.sum(pw_probs)

        # pw_probs = np.triu(pw_probs, k=1)
        pw_probs.retain_grad()  # debug: view the backward pass result

        return sdp_obj_value, pw_probs

    def build_trellis(self, pw_probs: np.ndarray, only_avg_hac: bool = False):
        t = Trellis(adj_mx=pw_probs)
        t.fit(only_avg_hac=only_avg_hac)
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
            if self.num_ecc > 0:
                num_ecc_sat[node] = self.get_num_ecc_sat(
                        leaves, self.num_points)
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
        if self.num_ecc > 0:
            best_clustering = best_clustering[:-self.num_ecc]

        return best_clustering, obj_vals[node], num_ecc_sat[node]

    def pred(self, only_avg_hac: bool = False):
        num_ecc = len(self.ecc_constraints)

        # Construct and solve SDP
        start_solve_time = time.time()
        sdp_obj_value, pw_probs = self.build_and_solve_sdp()
        end_solve_time = time.time()

        # # Build trellis
        # t = self.build_trellis(pw_probs, only_avg_hac=only_avg_hac)
        #
        # # Cut trellis
        # pred_clustering, cut_obj_value, num_ecc_satisfied = self.cut_trellis(t)

        # Forward pass through the trellis-cut rounding procedure
        rounded_solution = self.rounding_layer(pw_probs, only_avg_hac=only_avg_hac)
        rounded_solution.retain_grad()  # debug: view the backward pass result
        gold_solution = torch.zeros(pw_probs.size()).random_(0, 2)  # dummy
        loss = torch.norm(gold_solution - rounded_solution)
        embed()


        metrics = {
                'sdp_solve_time': end_solve_time - start_solve_time,
                'sdp_obj_value': sdp_obj_value,
                'cut_obj_value': self.rounding_layer.cut_obj_value,
                'num_ecc_satisfied': int(self.rounding_layer.num_ecc_satisfied),
                'num_ecc': num_ecc,
                'frac_ecc_satisfied': self.rounding_layer.num_ecc_satisfied / num_ecc
                        if num_ecc > 0 else 0.0,
                'num_ecc_feats': self.ecc_mx.nnz 
                        if self.ecc_mx is not None else 0,
                'num_pos_ecc_feats': (self.ecc_mx > 0).nnz
                        if self.ecc_mx is not None else 0,
                'num_neg_ecc_feats': (self.ecc_mx < 0).nnz
                        if self.ecc_mx is not None else 0,
        }

        return self.rounding_layer.pred_clustering, metrics


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


@nb.njit
def argmaximin(row_max: np.ndarray,
               col_max: np.ndarray,
               row_argmax: np.ndarray,
               col_argmax: np.ndarray,
               row_indices: np.ndarray,
               col_indices: np.ndarray):
    # This function picks the pair of gold and pred clusters which has 
    # the highest potential gain in cluster F1.
    best_maximin = 1.0
    best_argmaximin = (0, 0)
    for i in range(row_indices.size):
        curr_maximin = max(row_max[row_indices[i]], col_max[col_indices[i]])
        if row_max[row_indices[i]] < col_max[col_indices[i]]:
            curr_argmaximin = (col_argmax[col_indices[i]], col_indices[i])
        else:
            curr_argmaximin = (row_indices[i], row_argmax[row_indices[i]])
        if curr_maximin < best_maximin:
            best_maximin = curr_maximin
            best_argmaximin = curr_argmaximin
    return best_argmaximin


@nb.njit
def nb_isin_sorted(values: np.ndarray, query: int):
    dom_min = 0             # inclusive
    dom_max = values.size   # exclusive
    while dom_max - dom_min > 0:
        i = ((dom_max - dom_min) // 2) + dom_min
        if values[i] > query:
            dom_max = i
        elif values[i] < query:
            dom_min = i + 1
        else:
            return True
    return False


@nb.njit
def get_salient_feats(point_feats_indptr: np.ndarray,
                      point_feats_indices: np.ndarray,
                      point_idxs: np.ndarray,
                      salient_feat_counts: np.ndarray):
    for i in range(point_feats_indptr.size-1):
        in_focus_set = nb_isin_sorted(point_idxs, i)
        for j in range(point_feats_indptr[i], point_feats_indptr[i+1]):
            if salient_feat_counts[point_feats_indices[j]] == -1:
                continue
            if not in_focus_set:
                salient_feat_counts[point_feats_indices[j]] = -1
            else:
                salient_feat_counts[point_feats_indices[j]] += 1


@nb.njit
def get_point_matching_mx(gold_cluster_lbls: np.ndarray,
                          pred_cluster_lbls: np.ndarray):
    num_gold_clusters = np.unique(gold_cluster_lbls).size
    num_pred_clusters = np.unique(pred_cluster_lbls).size
    intersect_mx = np.zeros((num_gold_clusters, num_pred_clusters))
    union_mx = np.zeros((num_gold_clusters, num_pred_clusters))
    for i in range(gold_cluster_lbls.size):
        gold_idx = gold_cluster_lbls[i]
        pred_idx = pred_cluster_lbls[i]
        intersect_mx[gold_idx, pred_idx] += 1
        union_mx[gold_idx, :] += 1
        union_mx[:, pred_idx] += 1
        union_mx[gold_idx, pred_idx] -= 1
    return intersect_mx / union_mx


def gen_forced_ecc_constraint(point_feats: csr_matrix,
                              gold_cluster_lbls: np.ndarray,
                              pred_cluster_lbls: np.ndarray,
                              gold_cluster_feats: csr_matrix,
                              pred_cluster_feats: csr_matrix,
                              matching_mx: np.ndarray,
                              max_pos_feats: int):

    # construct the point matching matrix
    point_matching_mx = get_point_matching_mx(
            gold_cluster_lbls, pred_cluster_lbls)

    # set perfect match rows and columns to zero so they will not be picked
    perfect_match = (point_matching_mx == 1.0)
    row_mask = np.any(perfect_match, axis=1)
    column_mask = np.any(perfect_match, axis=0)
    to_zero_mask = row_mask[:, None] | column_mask[None, :]
    point_matching_mx[to_zero_mask] = 0.0

    # greedily pick minimax match
    row_max = np.max(point_matching_mx, axis=1)
    col_max = np.max(point_matching_mx, axis=0)
    row_argmax = np.argmax(point_matching_mx, axis=1)
    col_argmax = np.argmax(point_matching_mx, axis=0)
    row_indices, col_indices = np.where(point_matching_mx > 0.0)
    gold_cluster_idx, pred_cluster_idx = argmaximin(
            row_max, col_max, row_argmax, col_argmax, row_indices, col_indices)

    logging.info(f'Gold Cluster: {gold_cluster_idx},'
                 f' Pred Cluster: {pred_cluster_idx}')
    
    # get points in gold and pred clusters, resp.
    gold_cluster_points = set(np.where(gold_cluster_lbls==gold_cluster_idx)[0])
    pred_cluster_points = set(np.where(pred_cluster_lbls==pred_cluster_idx)[0])

    gold_and_pred = np.asarray(list(gold_cluster_points & pred_cluster_points))
    gold_not_pred = np.asarray(list(gold_cluster_points - pred_cluster_points))
    pred_not_gold = np.asarray(list(pred_cluster_points - gold_cluster_points))

    # start the sampling process with overlap feats
    gold_and_pred_sfc = np.zeros((point_feats.shape[1],))
    get_salient_feats(
            point_feats.indptr,
            point_feats.indices,
            np.sort(gold_and_pred),
            gold_and_pred_sfc
    )
    #sampled_overlap_feats = np.where(gold_and_pred_sfc == 1.0)[0][:1]
    # NOTE: why doesn't this line below work well with the SDP?
    # i.e. why don't the most common features work best
    sampled_overlap_feats = np.argsort(gold_and_pred_sfc)[-1:]

    # now onto postive feats
    sampled_pos_feats = []
    gold_not_pred_lbls = np.asarray(
            [pred_cluster_lbls[i] for i in gold_not_pred])
    for pred_lbl in np.unique(np.asarray(gold_not_pred_lbls))[:max_pos_feats-1]:
        pred_cluster_mask = (gold_not_pred_lbls == pred_lbl)
        gold_not_pred_sfc = np.zeros((point_feats.shape[1],))
        get_salient_feats(
                point_feats.indptr,
                point_feats.indices,
                np.sort(gold_not_pred[pred_cluster_mask]),
                gold_not_pred_sfc
        )
        sampled_pos_feats.append(np.argmax(gold_not_pred_sfc))
    sampled_pos_feats = np.asarray(sampled_pos_feats)

    # lastly, negative feats
    sampled_neg_feats = []
    pred_not_gold_lbls = np.asarray(
            [gold_cluster_lbls[i] for i in pred_not_gold])
    for gold_lbl in np.unique(np.asarray(pred_not_gold_lbls)):
        pred_cluster_mask = (pred_not_gold_lbls == gold_lbl)
        pred_not_gold_sfc = np.zeros((point_feats.shape[1],))
        get_salient_feats(
                point_feats.indptr,
                point_feats.indices,
                np.sort(pred_not_gold[pred_cluster_mask]),
                pred_not_gold_sfc
        )
        sampled_neg_feats.append(np.argmax(pred_not_gold_sfc))
    sampled_neg_feats = np.asarray(sampled_neg_feats)

    # create the ecc constraint
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

    new_ecc = coo_matrix(
            (new_ecc_data, (new_ecc_row, new_ecc_col)),
            shape=(1,point_feats.shape[1]),
            dtype=np.int64
    ).tocsr()

    # for debugging
    constraint_str = ', '.join(
            [('+f' if d > 0 else '-f') + str(int(c))
                for c, d in zip(new_ecc_col, new_ecc_data)]
    )
    logging.info(f'Constraint generated: [{constraint_str}]')

    logging.info('Nodes with features: {')
    for feat_id in new_ecc_col:
        nodes_with_feat = point_feats.T[feat_id].tocoo().col
        nodes_with_feat = [f'n{i}' for i in nodes_with_feat]
        logging.info(f'\tf{int(feat_id)}: {", ".join(nodes_with_feat)}')
    logging.info('}')

    # generate "equivalent" pairwise point constraints
    overlap_feats = set(sampled_overlap_feats)
    pos_feats = set(sampled_pos_feats)
    neg_feats = set(sampled_neg_feats)

    gold_not_pred = gold_cluster_points - pred_cluster_points
    pred_not_gold = pred_cluster_points - gold_cluster_points

    num_points = point_feats.shape[0]
    pairwise_constraints = dok_matrix((num_points, num_points))
    for s, t in product(pred_cluster_points, gold_not_pred):
        s_feats = set(point_feats[s].indices)
        t_feats = set(point_feats[t].indices)
        if not (s_feats.isdisjoint(overlap_feats) 
                or t_feats.isdisjoint(pos_feats)):
            if gold_cluster_lbls[s] == gold_cluster_lbls[t]:
                if s < t:
                    pairwise_constraints[s, t] = 1
                else:
                    pairwise_constraints[t, s] = 1

    for s, t in product(gold_cluster_points, pred_not_gold):
        s_feats = set(point_feats[s].indices)
        t_feats = set(point_feats[t].indices)
        if not (s_feats.isdisjoint(overlap_feats) 
                or t_feats.isdisjoint(neg_feats)):
            if gold_cluster_lbls[s] != gold_cluster_lbls[t]:
                if s < t:
                    pairwise_constraints[s, t] = -1
                else:
                    pairwise_constraints[t, s] = -1

    return new_ecc, pairwise_constraints.tocoo()


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
            p = col_wt[choices]
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

    gold_not_pred = gold_cluster_points - pred_cluster_points
    pred_not_gold = pred_cluster_points - gold_cluster_points

    num_points = point_feats.shape[0]
    pairwise_constraints = dok_matrix((num_points, num_points))
    for s, t in product(pred_cluster_points, gold_not_pred):
        s_feats = set(point_feats[s].indices)
        t_feats = set(point_feats[t].indices)
        if not (s_feats.isdisjoint(overlap_feats) 
                or t_feats.isdisjoint(pos_feats)):
            if gold_cluster_lbls[s] == gold_cluster_lbls[t]:
                if s < t:
                    pairwise_constraints[s, t] = 1
                else:
                    pairwise_constraints[t, s] = 1

    for s, t in product(gold_cluster_points, pred_not_gold):
        s_feats = set(point_feats[s].indices)
        t_feats = set(point_feats[t].indices)
        if not (s_feats.isdisjoint(overlap_feats) 
                or t_feats.isdisjoint(neg_feats)):
            if gold_cluster_lbls[s] != gold_cluster_lbls[t]:
                if s < t:
                    pairwise_constraints[s, t] = -1
                else:
                    pairwise_constraints[t, s] = -1

    return new_ecc, pairwise_constraints.tocoo()


def simulate(edge_weights: csr_matrix,
             point_features: csr_matrix,
             gold_clustering: np.ndarray,
             max_rounds: int,
             max_sdp_iters: int,
             max_pos_feats: int,
             only_avg_hac: bool = False,
             max_num_ecc: int = -1):
    if max_num_ecc == -1:
        max_num_ecc = max_rounds

    gold_cluster_feats = sp_vstack([
        get_cluster_feats(point_features[gold_clustering == i])
            for i in np.unique(gold_clustering)
    ])

    clusterer = EccClusterer(edge_weights=edge_weights,
                             features=point_features,
                             max_num_ecc=max_num_ecc,  # max_rounds
                             max_pos_feats=max_pos_feats,
                             max_sdp_iters=max_sdp_iters)

    pairwise_constraints_for_replay = []
    round_pred_clusterings = []

    for r in range(max_rounds):
        # compute predicted clustering
        pred_clustering, metrics = clusterer.pred(only_avg_hac=only_avg_hac)

        # get predicted cluster feats and do some label remapping for later
        uniq_pred_cluster_lbls = np.unique(pred_clustering)
        pred_cluster_feats = sp_vstack([
            get_cluster_feats(point_features[pred_clustering == i])
                for i in uniq_pred_cluster_lbls
        ])
        remap_lbl_dict = {j: i for i, j in enumerate(uniq_pred_cluster_lbls)}
        for i in range(pred_clustering.size):
            pred_clustering[i] = remap_lbl_dict[pred_clustering[i]]

        # for debugging
        logging.info('Gold Clustering: {')
        for cluster_id in np.unique(gold_clustering):
            nodes_in_cluster = list(np.where(gold_clustering == cluster_id)[0])
            nodes_in_cluster = [f'n{i}' for i in nodes_in_cluster]
            logging.info(f'\tc{cluster_id}: {", ".join(nodes_in_cluster)}')
        logging.info('}')

        logging.info('Predicted Clustering: {')
        for cluster_id in np.unique(pred_clustering):
            nodes_in_cluster = list(np.where(pred_clustering == cluster_id)[0])
            nodes_in_cluster = [f'n{i}' for i in nodes_in_cluster]
            logging.info(f'\tc{cluster_id}: {", ".join(nodes_in_cluster)}')
        logging.info('}')

        # record `pred_clustering` for later analysis
        round_pred_clusterings.append(copy.deepcopy(pred_clustering))

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
        metrics['f1'] = cluster_f1(gold_clustering, pred_clustering)[2]
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

        if r < min(max_rounds, max_num_ecc) - 1:
            # generate a new constraint
            while True:
                ecc_constraint, pairwise_constraints = gen_forced_ecc_constraint(
                        point_features,
                        gold_clustering,
                        pred_clustering,
                        gold_cluster_feats,
                        pred_cluster_feats,
                        matching_mx,
                        max_pos_feats
                )
                already_exists = any([
                    (ecc_constraint != x).nnz == 0
                        for x in clusterer.ecc_constraints
                ])
                if already_exists:
                    logging.error('Produced duplicate ecc constraint')
                    exit()
                    continue

                already_satisfied = (
                    (pred_cluster_feats @ ecc_constraint.T) == ecc_constraint.nnz
                ).todense().any()
                if already_satisfied:
                    logging.warning('Produced already satisfied ecc constraint')
                    continue

                pairwise_constraints_for_replay.append(pairwise_constraints)
                break

            logging.info('Adding new constraint')
            clusterer.add_constraint(ecc_constraint)

    return (clusterer.ecc_constraints,
            pairwise_constraints_for_replay,
            round_pred_clusterings)


def get_hparams():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--debug', action='store_true',
                        help="Enables and disables certain opts for debugging")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output directory for this run.")
    parser.add_argument("--data_path", default=None, type=str, required=True,
                        help="Path to preprocessed data.")

    parser.add_argument('--max_rounds', type=int, default=100,
                        help="number of rounds to generate feedback for")
    parser.add_argument('--max_sdp_iters', type=int, default=50000,
                        help="max num iterations for sdp solver")

    # for constraint generation
    parser.add_argument('--max_pos_feats', type=int, default=6,
                        help="max number of positive feats in ecc constraint.")

    # for end-to-end training
    parser.add_argument('--only_avg_hac', action='store_true',
                        help="Builds only the average linkage tree instead of a trellis of 5 trees")
    parser.add_argument('--max_num_ecc', type=int, default=-1,
                        help="Maximum number of existential cluster constraints. -1 defaults to using `max_rounds`")

    hparams = parser.parse_args()
    return hparams


if __name__ == '__main__':

    hparams = get_hparams()

    if not hparams.debug:
        # create output directory
        assert not os.path.exists(hparams.output_dir)
        os.makedirs(hparams.output_dir)

        # dump hparams
        pickle.dump(
                hparams, 
                open(os.path.join(hparams.output_dir, 'hparams.pkl'), "wb")
        )
        logging.basicConfig(
                format='(ECC) :: %(asctime)s >> %(message)s',
                datefmt='%m-%d-%y %H:%M:%S',
                level=logging.INFO,
                handlers=[
                    logging.FileHandler(os.path.join(hparams.output_dir, "out.log")),
                    logging.StreamHandler()
                ],
        )
    else:
        logging.basicConfig(
                format='(ECC) :: %(asctime)s >> %(message)s',
                datefmt='%m-%d-%y %H:%M:%S',
                level=logging.INFO
        )

    logging.info('Experiment args:\n{}'.format(
        json.dumps(vars(hparams), sort_keys=True, indent=4)))

    pl.utilities.seed.seed_everything(hparams.seed)

    with open(hparams.data_path, 'rb') as f:
        logging.info('Loading preprocessed data.')
        blocks_preprocessed = pickle.load(f)

    ecc_for_replay = {}
    mlcl_for_replay = {}
    pred_clusterings = {}
    num_blocks = len(blocks_preprocessed)

    #sub_blocks_preprocessed = {}
    #sub_blocks_preprocessed['d schmidt'] = blocks_preprocessed['d schmidt']
    sub_blocks_preprocessed = blocks_preprocessed

    for i, (block_name, block_data) in enumerate(sub_blocks_preprocessed.items()):
        edge_weights = block_data['edge_weights']
        point_features = block_data['point_features']
        gold_clustering = block_data['labels']

        assert edge_weights.shape[0] == point_features.shape[0]
        num_clusters = np.unique(gold_clustering).size

        logging.info(f'Loaded block \"{block_name}\" ({i+1}/{num_blocks})')
        logging.info(f'\t number of points: {edge_weights.shape[0]}')
        logging.info(f'\t number of clusters: {num_clusters}')
        logging.info(f'\t number of features: {point_features.shape[1]}')

        (block_ecc_for_replay,
         block_mlcl_for_replay,
         round_pred_clusterings) = simulate(
                edge_weights,
                point_features,
                gold_clustering,
                hparams.max_rounds,
                hparams.max_sdp_iters,
                hparams.max_pos_feats,
                only_avg_hac=hparams.only_avg_hac,
                max_num_ecc=hparams.max_num_ecc
        )

        ecc_for_replay[block_name] = block_ecc_for_replay
        mlcl_for_replay[block_name] = block_mlcl_for_replay
        pred_clusterings[block_name] = round_pred_clusterings

    if not hparams.debug:
        logging.info('Dumping ecc and mlcl constraints for replay')
        ecc_fname = os.path.join(hparams.output_dir, 'ecc_for_replay.pkl')
        mlcl_fname = os.path.join(hparams.output_dir, 'mlcl_for_replay.pkl')
        pred_fname = os.path.join(hparams.output_dir, 'pred_clusterings.pkl')
        pickle.dump(ecc_for_replay, open(ecc_fname, 'wb'))
        pickle.dump(mlcl_for_replay, open(mlcl_fname, 'wb'))
        pickle.dump(pred_clusterings, open(pred_fname, 'wb'))
