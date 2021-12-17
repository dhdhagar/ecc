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

    def solve_sdp(self):
        num_points = self.features.shape[0]
        num_ecc = len(self.ecc_constraints)

        logging.info('Constructing optimization problem')
        W = csr_matrix((self.edge_weights.data,
                        (self.edge_weights.row, self.edge_weights.col)),
                        shape=(self.n, self.n))
        ecc_mx = sp_vstack(self.ecc_constraints).T.tocsc()
        uni_feats = sp_vstack([self.features, ecc_mx])

        embed()
        exit()

        X = cp.Variable((self.n, self.n), PSD=True)
        constraints = [
                cp.diag(X) == np.ones((self.n,)),
                X >= 0
        ]

        # TODO: build out incompatibility and satisfiability sdp constraints
        embed()
        exit()

        prob = cp.Problem(cp.Maximize(cp.trace(W @ X)), constraints)

        logging.info('Solving optimization problem')
        prob.solve(solver=cp.SCS, verbose=True)
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

    def cut_trellis(self, t: hg.Tree):
        parents = t.parents()
        membership = copy.deepcopy(parents[:self.n])
        best_clustering = np.arange(self.n)
        obj_vals = np.zeros((2*self.n - 1,))
        num_ecc_sat = np.zeros((2*self.n - 1,))

        for node in t.leaves_to_root_iterator(
                include_leaves=False, include_root=True):
            leaves_mask = (membership == node)
            leaves = np.where(leaves_mask)[0]
            obj_vals[node] = self.get_intra_cluster_energy(leaves)
            curr_obj_val = sum([obj_vals[i] 
                for i in np.unique(best_clustering[leaves_mask])])
            if obj_vals[node] > curr_obj_val:
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
