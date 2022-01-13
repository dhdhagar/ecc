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
from sklearn.metrics import homogeneity_completeness_v_measure as cluster_f1

from trellis import Trellis

from IPython import embed


class MLCLClusterer(object):

    def __init__(self,
                 edge_weights: csr_matrix,
                 max_sdp_iters: int):

        self.edge_weights = copy.deepcopy(edge_weights)
        self.large_num = np.sum(np.abs(edge_weights.data))
        self.max_sdp_iters = max_sdp_iters
        self.n = self.edge_weights.shape[0]
        self.mlcl_constraints = []

    def add_constraint(self, mlcl_constraint: Tuple[int, int, int]):
        i, j, s = mlcl_constraint
        assert s in [-1, 1]
        self.edge_weights[i, j] = s * self.large_num
        self.mlcl_constraints.append(mlcl_constraint)

    def build_and_solve_sdp(self):
        # formulate SDP
        logging.info('Constructing optimization problem')

        W = self.edge_weights
        X = cp.Variable((self.n, self.n), PSD=True)
        # standard correlation clustering constraints
        constraints = [
                cp.diag(X) == np.ones((self.n,)),
                X >= 0
        ]
        ## add must-link and cannot-link constraints
        #for i, j, s in self.mlcl_constraints:
        #    if s == 1:
        #        constraints.append(X[i,j] >= 1)
        #        constraints.append(X[j,i] >= 1)
        #    elif s == -1:
        #        constraint.append(X[i,j] <= 0)
        #        constraint.append(X[j,i] <= 0)
        #    else:
        #        raise ValueError('Invalid sign value in mlcl constraint.')
        
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
        # convert to coo for this function
        self.edge_weights = self.edge_weights.tocoo()

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

        # make sure all mlcl constraints are satisfied
        for i, j, s in self.mlcl_constraints:
            if s == 1:
                assert best_clustering[i] == best_clustering[j]
            else:
                assert best_clustering[i] != best_clustering[j]

        # convert back to csr for the rest of the class
        self.edge_weights = self.edge_weights.tocsr()

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
    pred_same_cluster = (pred_cluster_lbls[:, None]
                         == pred_cluster_lbls[None, :])
    gold_same_cluster = (gold_cluster_lbls[:, None]
                         == gold_cluster_lbls[None, :])
    incorrect_pairs = (pred_same_cluster != gold_same_cluster)
    incorrect_pairs = np.triu(incorrect_pairs, k=1).astype(float)
    pair_probs = incorrect_pairs / np.sum(incorrect_pairs)
    pair_ravel_idx = np.where(np.random.multinomial(1, pair_probs.ravel()))
    i, j = np.unravel_index(pair_ravel_idx[0][0], pair_probs.shape)

    if gold_cluster_lbls[i] == gold_cluster_lbls[j]:
        mlcl_constraint = (i, j, 1)
        ecc_constraint = (point_feats[i] + point_feats[j])
        ecc_constraint.data = np.clip(ecc_constraint.data, 0, 1)
        ecc_constraints = [ecc_constraint]
    else:
        mlcl_constraint = (i, j, -1)
        ecc_constraint1 = (point_feats[i] - point_feats[j]) + point_feats[i]
        ecc_constraint1.data = np.clip(ecc_constraint1.data, -1, 1)
        ecc_constraint2 = (point_feats[j] - point_feats[i]) + point_feats[j]
        ecc_constraint2.data = np.clip(ecc_constraint2.data, -1, 1)
        ecc_constraints = [ecc_constraint1, ecc_constraint2]

    return mlcl_constraint, ecc_constraints


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


def gen_forced_mlcl_constraints(point_feats: csr_matrix,
                                gold_cluster_lbls: np.ndarray,
                                pred_cluster_lbls: np.ndarray,
                                matching_mx: np.ndarray):

    mlcl_constraints = []

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

    largest_potential = 0

    # now onto postive feats
    sampled_pos_feats = []
    gold_not_pred_lbls = np.asarray(
            [pred_cluster_lbls[i] for i in gold_not_pred])
    for pred_lbl in np.unique(np.asarray(gold_not_pred_lbls)):
        pred_cluster_mask = (gold_not_pred_lbls == pred_lbl)
        edge = (gold_and_pred[0], gold_not_pred[pred_cluster_mask][0])
        if edge[0] < edge[1]:
            mlcl_constraints.append((edge[0], edge[1], 1))
        else:
            mlcl_constraints.append((edge[1], edge[0], 1))

        if np.sum(pred_cluster_mask) > largest_potential:
            largest_potential = np.sum(pred_cluster_mask)
            mlcl_constraints[0], mlcl_constraints[-1] = mlcl_constraints[-1], mlcl_constraints[0]

    # lastly, negative feats
    sampled_neg_feats = []
    pred_not_gold_lbls = np.asarray(
            [gold_cluster_lbls[i] for i in pred_not_gold])
    for gold_lbl in np.unique(np.asarray(pred_not_gold_lbls)):
        pred_cluster_mask = (pred_not_gold_lbls == gold_lbl)
        edge = (gold_and_pred[0], pred_not_gold[pred_cluster_mask][0])
        if edge[0] < edge[1]:
            mlcl_constraints.append((edge[0], edge[1], -1))
        else:
            mlcl_constraints.append((edge[1], edge[0], -1))

        if np.sum(pred_cluster_mask) > largest_potential:
            largest_potential = np.sum(pred_cluster_mask)
            mlcl_constraints[0], mlcl_constraints[-1] = mlcl_constraints[-1], mlcl_constraints[0]

    return mlcl_constraints


def simulate(edge_weights: csr_matrix,
             point_features: csr_matrix,
             gold_clustering: np.ndarray,
             max_rounds: int,
             max_sdp_iters: int,
             gen_constraints_in_batches: bool): 

    gold_cluster_feats = sp_vstack([
        get_cluster_feats(point_features[gold_clustering == i])
            for i in np.unique(gold_clustering)
    ])

    clusterer = MLCLClusterer(edge_weights=edge_weights,
                              max_sdp_iters=max_sdp_iters)

    round_pred_clusterings = []

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

        # generate a new constraint
        mlcl_constraints = gen_forced_mlcl_constraints(
                point_features,
                gold_clustering,
                pred_clustering,
                matching_mx
        )
        if gen_constraints_in_batches:
            logging.info('Adding new constraints')
            for mlcl_constraint in mlcl_constraints:
                clusterer.add_constraint(mlcl_constraint)
                logging.info(f'n{mlcl_constraint[0]} {"-" if mlcl_constraint[2] < 0 else "+"} n{mlcl_constraint[1]}')
        else:
            logging.info('Adding new constraint')
            clusterer.add_constraint(mlcl_constraints[0])
            logging.info(f'n{mlcl_constraints[0][0]} {"-" if mlcl_constraints[0][2] < 0 else "+"} n{mlcl_constraints[0][1]}')

    return clusterer.mlcl_constraints, round_pred_clusterings


def get_hparams():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--debug', action='store_true',
                        help="Enables and disables certain opts for debugging")
    parser.add_argument('--gen_constraints_in_batches', action='store_true',
                        help="Whether or not to generate more than one"
                             " constraint at a time.")
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

    mlcl_for_replay = {}
    pred_clusterings = {}
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

        block_mlcl_for_replay, round_pred_clusterings = simulate(
                edge_weights,
                point_features,
                gold_clustering,
                hparams.max_rounds,
                hparams.max_sdp_iters,
                hparams.gen_constraints_in_batches
        )

        mlcl_for_replay[block_name] = block_mlcl_for_replay
        pred_clusterings[block_name] = round_pred_clusterings

    if not hparams.debug:
        logging.info('Dumping ecc and mlcl constraints for replay')
        mlcl_fname = os.path.join(hparams.output_dir, 'mlcl_for_replay.pkl')
        pred_fname = os.path.join(hparams.output_dir, 'pred_clusterings.pkl')
        pickle.dump(mlcl_for_replay, open(mlcl_fname, 'wb'))
        pickle.dump(pred_clusterings, open(pred_fname, 'wb'))
