import copy
import heapq
import logging
import pickle
from typing import List

import higra as hg
import numba as nb
import numpy as np
from sklearn.metrics import adjusted_rand_score as rand_idx

from IPython import embed


class Trellis(object):

    def __init__(self, adj_mx: np.ndarray, beam_width: int):
        self.adj_mx = adj_mx
        self.beam_width = beam_width
        self.n = adj_mx.shape[0]

    def get_top_beam_merges(self, membership: np.ndarray):
        node_frontier = np.unique(membership)
        h = [] # heap for storing top beam merges
        for idx, n1 in enumerate(node_frontier):
            for n2 in node_frontier[idx+1:]:
                merge_mask = (np.isin(membership, [n1])[:, None]
                              & np.isin(membership, [n2])[None, :])
                energy = np.mean(self.adj_mx[merge_mask])
                if len(h) < self.beam_width:
                    heapq.heappush(h, (energy, (n1, n2)))
                else:
                    # this automatically pops the least desirable merge so
                    # far and drops it.
                    heapq.heappushpop(h, (energy, (n1, n2)))
        return h


    def beam_search(self) -> List[np.ndarray]:
        num_tree_nodes = (2*self.n) - 1
        parent = np.empty((self.beam_width, num_tree_nodes), dtype=np.uint32)
        membership = np.tile(np.arange(self.n, dtype=np.uint32),
                             (self.beam_width, 1))

        for curr in range(self.n, num_tree_nodes):
            beam_heap = []
            for b in range(self.beam_width):
                top_merges = self.get_top_beam_merges(membership[b])
                for energy, merge_pair in top_merges:
                    prop_membership = copy.deepcopy(membership[b])
                    merge_mask = np.isin(prop_membership, merge_pair)
                    prop_membership[merge_mask] = curr
                    already_exists = False
                    if curr < num_tree_nodes - 1:
                        for (_, _, (_, _, other_membership, _)) in beam_heap:
                            if rand_idx(prop_membership, other_membership) == 1:
                                already_exists = True
                                break
                    if not already_exists:
                        prop_parent = copy.deepcopy(parent[b])
                        prop_parent[merge_pair[0]] = curr
                        prop_parent[merge_pair[1]] = curr
                        cand_tuple = (energy, b, (merge_pair[0],
                                                  merge_pair[1],
                                                  prop_membership,
                                                  prop_parent))
                        if len(beam_heap) < self.beam_width:
                            heapq.heappush(beam_heap, cand_tuple)
                        else:
                            # this automatically pops the least desirable
                            # merge so far and drops it.
                            heapq.heappushpop(beam_heap, cand_tuple)
            
            # update membership and parent with latest beams
            for b, (_, _, (_, _, m, p)) in enumerate(beam_heap):
                parent[b] = p
                membership[b] = m

        # the root is always its own parent
        parent[:,-1] = (num_tree_nodes - 1)
        return parent


    @staticmethod
    @nb.njit(nogil=True)
    def get_trellis_node_id(leaves_indptr: np.ndarray,
                            leaves_indices: np.ndarray,
                            new_leaves: np.ndarray):
        num_trellis_nodes = leaves_indptr.size - 1
        num_new_leaves = new_leaves.size
        node_id = -1
        for i in range(num_trellis_nodes):
            ptr = leaves_indptr[i]
            next_ptr = leaves_indptr[i+1]
            if (next_ptr - ptr) == num_new_leaves:
                match = True
                for idx, j in enumerate(range(ptr, next_ptr)):
                    if leaves_indices[j] != new_leaves[idx]:
                        match = False
                        break
                if match:
                    node_id = i
                    break
        if node_id == -1:
            # node does not exist yet; create new trellis node in `leaves_*`
            node_id = num_trellis_nodes
            leaves_indptr = np.append(
                    leaves_indptr, leaves_indptr[-1] + num_new_leaves)
            leaves_indices = np.append(leaves_indices, new_leaves)
        return node_id, leaves_indptr, leaves_indices

    @staticmethod
    @nb.njit(nogil=True, parallel=True)
    def build_trellis_from_trees(trees: np.ndarray):
        num_trees = trees.shape[0]
        num_tree_nodes = trees.shape[1]
        num_leaves = (num_tree_nodes + 1) // 2
        leaves_indptr = np.arange(num_leaves+1, dtype=np.int64)
        leaves_indices = np.arange(num_leaves, dtype=np.int64)
        child_pairs_indptr = np.zeros((num_leaves+1,), dtype=np.int64)
        child_pairs_indices = np.array([], dtype=np.int64)

        for b in range(num_trees):
            membership = np.arange(num_leaves)
            for curr in range(num_leaves, num_tree_nodes):
                child_mask = (trees[b] == curr)
                children = np.where(child_mask)[0]
                curr_leaves_mask = np.zeros_like(membership, dtype=bool)
                lchild_leaves_mask = np.zeros_like(membership, dtype=bool)
                rchild_leaves_mask = np.zeros_like(membership, dtype=bool)
                for i in nb.prange(num_leaves):
                    if membership[i] == children[0]:
                        lchild_leaves_mask[i] = True
                        curr_leaves_mask[i] = True
                    elif membership[i] == children[1]:
                        rchild_leaves_mask[i] = True
                        curr_leaves_mask[i] = True
                membership[curr_leaves_mask] = curr
                curr_leaves = np.where(curr_leaves_mask)[0]
                lchild_leaves = np.where(lchild_leaves_mask)[0]
                rchild_leaves = np.where(rchild_leaves_mask)[0]

                prev_num_trellis_nodes = leaves_indptr.size - 1
                (lchild_node_id,
                 leaves_indptr,
                 leaves_indices) = Trellis.get_trellis_node_id(
                        leaves_indptr, leaves_indices, lchild_leaves)
                (rchild_node_id,
                 leaves_indptr,
                 leaves_indices) = Trellis.get_trellis_node_id(
                        leaves_indptr, leaves_indices, rchild_leaves)
                assert lchild_node_id < prev_num_trellis_nodes
                assert rchild_node_id < prev_num_trellis_nodes

                (curr_node_id,
                 leaves_indptr,
                 leaves_indices) = Trellis.get_trellis_node_id(
                        leaves_indptr, leaves_indices, curr_leaves)
                assert curr_node_id <= prev_num_trellis_nodes

                if curr_node_id == prev_num_trellis_nodes:
                    child_pairs_indptr = np.append(
                            child_pairs_indptr, child_pairs_indptr[-1]+2)
                    child_pairs_indices = np.append(
                            child_pairs_indices,
                            [lchild_node_id, rchild_node_id])
                else:
                    ptr = child_pairs_indptr[curr_node_id]
                    next_ptr = child_pairs_indptr[curr_node_id+1]
                    already_exists = False
                    for i in range(ptr, next_ptr):
                        if child_pairs_indices[i] == lchild_node_id:
                            assert child_pairs_indices[i+1] == rchild_node_id
                            already_exists = True
                            break
                    if not already_exists:
                        child_pairs_indptr[curr_node_id+1:] += 2
                        child_pairs_indices = np.concatenate((
                                child_pairs_indices[:next_ptr],
                                [lchild_node_id, rchild_node_id],
                                child_pairs_indices[next_ptr:]
                        ))

        return (leaves_indptr,
                leaves_indices,
                child_pairs_indptr,
                child_pairs_indices)

    def fit(self):
        # get the HAC tree, not necessarily contained in beam search
        g, w = hg.adjacency_matrix_2_undirected_graph(self.adj_mx)
        hac_tree, _ = hg.binary_partition_tree_average_linkage(g, -1.0*w)
        hac_tree = hac_tree.parents()[None, :]

        # run beam search to find top-`b` trees
        beam_trees = self.beam_search()

        # build trellis
        trees = np.concatenate((hac_tree, beam_trees), axis=0)
        (self.leaves_indptr,
         self.leaves_indices,
         self.child_pairs_indptr,
         self.child_pairs_indices) = self.build_trellis_from_trees(trees)

        # TODO: last thing is to set order for internal trellis node iteration
        embed()                                                     
        exit()



if __name__ == '__main__':
    logging.basicConfig(
            format='(trellis) :: %(asctime)s >> %(message)s',
            datefmt='%m-%d-%y %H:%M:%S',
            level=logging.INFO
    )

    with open('test_sim_graph.pkl', 'rb') as f:
        X = pickle.load(f)

    trellis = Trellis(adj_mx=X, beam_width=3)
    trellis.fit()
