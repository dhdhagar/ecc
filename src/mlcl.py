# Run ecc simualtion using synthetically generated must-link and cannot-link
# constraints.
#
# Usage:
#       python mlcl.py
#

import argparse
import copy
from itertools import product
import json
import logging
import os
import pickle
import time
from typing import Tuple

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


class MLCLClusterer(object):

    def __init__(self,
                 edge_weights: csr_matrix,
                 max_sdp_iters: int):

        self.edge_weights = edge_weights.tocoo()
        self.max_sdp_iters = max_sdp_iters
        self.n = self.edge_weights.shape[0]
        self.mlcl_constraints = []

    def add_constraint(self, mlcl_constraint: Tuple[int, int, int]):
        self.mlcl_constraints.append(mlcl_constraint)

    def build_and_solve_sdp(self):
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
        # add must-link and cannot-link constraints
        for i, j, s in self.mlcl_constraints:
            if s == 1:
                constraints.append(X[i,j] >= 1)
            elif s == -1:
                constraint.append(X[i,j] <= 0)
            else:
                raise ValueError('Invalid sign value in mlcl constraint.')
        
        prob = cp.Problem(cp.Maximize(cp.trace(W @ X)), constraints)

        logging.info('Solving optimization problem')
        sdp_obj_value = prob.solve(
                solver=cp.SCS, verbose=True, max_iters=self.max_sdp_iters
        )

        pw_probs = np.triu(X.value, k=1)
        return sdp_obj_value, pw_probs

    def build_trellis(self, pw_probs: np.ndarray):
        t = Trellis(adj_mx=pw_probs)
        t.fit()
        return t

    def get_intra_cluster_energy(self, leaves: np.ndarray):
        row_mask = np.isin(self.edge_weights.row, leaves)
        col_mask = np.isin(self.edge_weights.col, leaves)
        data_mask = row_mask & col_mask
        return np.sum(self.edge_weights.data[data_mask])

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

        for node in t.internal_nodes_topo_ordered():
            node_start = membership_indptr[node]
            node_end = membership_indptr[node+1]
            leaves = membership_indices[node_start:node_end]
            obj_vals[node] = self.get_intra_cluster_energy(leaves)
            for lchild, rchild in t.get_child_pairs_iter(node):
                cpair_obj_val = obj_vals[lchild] + obj_vals[rchild]
                if obj_vals[node] < cpair_obj_val:
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

        # TODO: make sure all mlcl constraints are satisfied

        return best_clustering, obj_vals[node]

    def pred(self):
        # Construct and solve SDP
        start_solve_time = time.time()
        sdp_obj_value, pw_probs = self.build_and_solve_sdp()
        end_solve_time = time.time()

        # Build trellis
        t = self.build_trellis(pw_probs)

        # Cut trellis
        pred_clustering, cut_obj_value = self.cut_trellis(t)

        metrics = {
                'sdp_solve_time': end_solve_time - start_solve_time,
                'sdp_obj_value': sdp_obj_value,
                'cut_obj_value': cut_obj_value,
                'num_mlcl': len(self.mlcl_constraints),
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

def gen_mlcl_constraint(gold_cluster_lbls: np.ndarray,
                        pred_cluster_lbls: np.ndarray,
                        point_feats: csr_matrix):
    pass


def simulate(edge_weights: csr_matrix,
             point_features: csr_matrix,
             gold_clustering: np.ndarray,
             max_rounds: int,
             max_sdp_iters: int): 

    clusterer = MLCLClusterer(edge_weights=edge_weights,
                              max_sdp_iters=max_sdp_iters)

    ecc_constraints_for_replay = []

    for r in range(max_rounds):
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
            mlcl_constraint, ecc_constraints = gen_mlcl_constraint(
                    gold_clustering,
                    pred_clustering,
                    point_features,
            )

            # TODO: compute `already_exists`

            if already_exists:
                raise RuntimeError('Generated constraint that already exists.')

            ecc_constraints_for_replay.append(ecc_constraints)
            break

        logging.info('Adding new constraint')
        clusterer.add_constraint(ecc_constraint)

    return ecc_constraints_for_replay, clusterer.mlcl_constraints


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
                filename=os.path.join(hparams.output_dir, 'out.log'),
                format='(MLCL) :: %(asctime)s >> %(message)s',
                datefmt='%m-%d-%y %H:%M:%S',
                level=logging.INFO
        )
    else:
        logging.basicConfig(
                format='(MLCL) :: %(asctime)s >> %(message)s',
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
    num_blocks = len(blocks_preprocessed)
    for i, (block_name, block_data) in enumerate(blocks_preprocessed.items()):
        edge_weights = block_data['edge_weights']
        point_features = block_data['point_features']
        gold_clustering = block_data['labels']

        assert edge_weights.shape[0] == point_features.shape[0]
        num_clusters = np.unique(gold_clustering).size

        logging.info(f'Loaded block \"{block_name}\" ({i+1}/{num_blocks})')
        logging.info(f'\t number of points: {edge_weights.shape[0]}')
        logging.info(f'\t number of clusters: {num_clusters}')
        logging.info(f'\t number of features: {point_features.shape[1]}')

        block_ecc_for_replay, block_mlcl_for_replay = simulate(
                edge_weights,
                point_features,
                gold_clustering,
                hparams.max_rounds,
                hparams.max_sdp_iters,
                hparams.max_overlap_feats,
                hparams.max_pos_feats,
                hparams.max_neg_feats
        )

        ecc_for_replay[block_name] = block_ecc_for_replay
        mlcl_for_replay[block_name] = block_mlcl_for_replay

    if not hparams.debug:
        logging.info('Dumping ecc and mlcl constraints for replay')
        ecc_fname = os.path.join(hparams.output_dir, 'ecc_for_replay.pkl')
        mlcl_fname = os.path.join(hparams.output_dir, 'mlcl_for_replay.pkl')
        pickle.dump(ecc_for_replay, open(ecc_fname, 'wb'))
        pickle.dump(mlcl_for_replay, open(mlcl_fname, 'wb'))
