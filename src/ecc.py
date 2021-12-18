# Run ecc simualtion using synthetically generated e-constraints.
#
# Usage:
#       python ecc.py
#

import copy
import logging
import pickle
import time

import cvxpy as cp
import higra as hg
import numba as nb
from numba.typed import List
import numpy as np
import pytorch_lightning as pl
from scipy.sparse import csr_matrix
from scipy.sparse import vstack as sp_vstack
from sklearn.metrics import adjusted_rand_score

from IPython import embed


class EccClusterer(object):

    def __init__(self, edge_weights: csr_matrix, features: np.ndarray):
        self.edge_weights = edge_weights.tocoo()
        self.features = features
        self.n = self.features.shape[0]
        self.ecc_constraints = []

    def add_constraint(self, ecc_constraint: csr_matrix):
        self.ecc_constraints.append(ecc_constraint)
        self.n += 1

    @staticmethod
    @nb.njit(nogil=True, parallel=True)
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
    @nb.njit(nogil=True)
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

    def solve_sdp(self):
        num_points = self.features.shape[0]
        num_ecc = len(self.ecc_constraints)

        # "negative" sdp constraints
        ecc_mx = sp_vstack(self.ecc_constraints)
        uni_feats = sp_vstack([self.features, ecc_mx])
        incompat_mx = np.zeros((num_points+num_ecc, num_ecc), dtype=bool)
        self._set_incompat_mx(num_points+num_ecc,
                              num_ecc,
                              uni_feats.indptr,
                              uni_feats.indices,
                              uni_feats.data,
                              ecc_mx.indptr,
                              ecc_mx.indices,
                              ecc_mx.data,
                              incompat_mx)
        ortho_indices = [(a, b+num_points)
                for a, b in zip(*np.where(incompat_mx))]

        # "positive" sdp constraints
        bin_features = self.features.astype(bool).tocsc()
        pos_ecc_mx = (ecc_mx > 0)
        (ecc_indices,
         points_indptr,
         points_indices) = self._get_feat_satisfied_hyperplanes(
                 bin_features.indptr,
                 bin_features.indices,
                 pos_ecc_mx.indptr,
                 pos_ecc_mx.indices,
                 incompat_mx)
        ecc_indices = [x + num_points for x in ecc_indices]
        points_indptr = list(points_indptr)
        points_indices = list(points_indices)

        # formulate SDP
        logging.info('Constructing optimization problem')
        W = csr_matrix((self.edge_weights.data,
                        (self.edge_weights.row, self.edge_weights.col)),
                        shape=(self.n, self.n))
        X = cp.Variable((self.n, self.n), PSD=True)
        # standard correlation clustering constraintsj
        constraints = [
                cp.diag(X) == np.ones((self.n,)),
                X >= 0
        ]
        # "negative" ecc constraints
        constraints.extend([X[i,j] <= 0 for i, j in ortho_indices])
        # "positive" ecc constraints
        for idx, i in enumerate(ecc_indices):
            j_s = points_indices[points_indptr[idx]: points_indptr[idx+1]]
            constraints.append(sum([X[i,j] for j in j_s]) >= 1)

        prob = cp.Problem(cp.Maximize(cp.trace(W @ X)), constraints)

        logging.info('Solving optimization problem')
        prob.solve(solver=cp.SCS, verbose=True, max_iters=2500)

        return np.triu(X.value, k=1)

    def build_trellis(self, X: np.ndarray):
        pp_graph, pp_edge_weights = hg.adjacency_matrix_2_undirected_graph(X)
        t, _ = hg.binary_partition_tree_average_linkage(
                pp_graph, 1.0-pp_edge_weights)
        return t

    def get_intra_cluster_energy(self, leaves: np.ndarray):
        row_mask = np.isin(self.edge_weights.row, leaves)
        col_mask = np.isin(self.edge_weights.col, leaves)
        data_mask = row_mask & col_mask
        return np.sum(self.edge_weights.data[data_mask])

    def get_num_ecc_sat(self, leaves: np.ndarray, num_points: int):
        embed()
        exit()

    def cut_trellis(self, t: hg.Tree):
        num_ecc = len(self.ecc_constraints)
        num_points = self.n - num_ecc
        parents = t.parents()
        membership = copy.deepcopy(parents[:self.n])
        best_clustering = np.arange(self.n)
        obj_vals = np.zeros((2*self.n - 1,))
        num_ecc_sat = np.zeros((2*self.n - 1,))

        for node in t.leaves_to_root_iterator(
                include_leaves=False, include_root=True):
            leaves_mask = (membership == node)
            leaves = np.where(leaves_mask)[0]
            num_ecc_sat[node] = self.get_num_ecc_sat(leaves, num_ecc)
            obj_vals[node] = self.get_intra_cluster_energy(leaves)
            curr_num_ecc_sat = sum([num_ecc_sat[i] 
                for i in np.unique(best_clustering[leaves_mask])])
            curr_obj_val = sum([obj_vals[i] 
                for i in np.unique(best_clustering[leaves_mask])])
            if (num_ecc_sat[node] > curr_num_ecc_sat 
                or (num_ecc_sat[node] == curr_num_ecc_sat
                    and obj_vals[node] > curr_obj_val)):
                best_clustering[leaves_mask] = node
            membership[leaves_mask] = parents[node]

        obj_value = sum([obj_vals[i] for i in np.unique(best_clustering)])
        num_ecc_satisfied = sum([num_ecc_sat[i]
            for i in np.unique(best_clustering)])

        return best_clustering, obj_value, num_ecc_satisfied

    def pred(self):
        # Construct and solve SDP
        start_solve_time = time.time()
        X_hat = self.solve_sdp()
        end_solve_time = time.time()

        # Build trellis
        t = self.build_trellis(X_hat)

        # Cut trellis
        pred_clustering, obj_value, num_ecc_satisfied = self.cut_trellis(t)

        metrics = {
                'sdp_solve_time': end_solve_time - start_solve_time,
                'obj_value': obj_value,
                'num_ecc_satisfied': num_ecc_satisfied,
        }

        return pred_clustering, metrics


def simulate(dc_graph: dict):
    edge_weights = dc_graph['edge_weights']
    point_features = dc_graph['point_features']
    cluster_features = dc_graph['cluster_features']
    gold_clustering = dc_graph['labels']

    ecc1 = csr_matrix(2*(cluster_features.todense()[0:1]) - 1)
    ecc2 = csr_matrix(2*(cluster_features.todense()[1:2]) - 1)

    clusterer = EccClusterer(edge_weights=edge_weights,
                             features=point_features)

    for r in range(10):
        clusterer.add_constraint(ecc1)
        clusterer.add_constraint(ecc2)
        pred_clustering, metrics = clusterer.pred()

        embed()
        exit()

        # TODO: compute metrics, break if clustering perfect
        # TODO: generate new ecc constraint


if __name__ == '__main__':
    seed = 42
    pl.utilities.seed.seed_everything(seed)

    data_fname = 'tiny_data.pkl'

    with open(data_fname, 'rb') as f:
        dc_graph = pickle.load(f)

    simulate(dc_graph)
