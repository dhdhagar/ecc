import copy
import heapq
import logging
import pickle
from typing import List

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
        cc_mask = (membership[:, None] == membership[None, :])
        h = [] # heap for storing top beam merges
        for idx, n1 in enumerate(node_frontier):
            for n2 in node_frontier[idx+1:]:
                merge_mask = cc_mask | (np.isin(membership, [n1])[:, None]
                        & np.isin(membership, [n2])[None, :])
                energy = np.sum(self.adj_mx[merge_mask])
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
                    for (_, (_, _, other_membership, _)) in beam_heap:
                        if rand_idx(prop_membership, other_membership) == 1:
                            already_exists = True
                            break
                    if not already_exists:
                        prop_parent = copy.deepcopy(parent[b])
                        prop_parent[merge_pair[0]] = curr
                        prop_parent[merge_pair[1]] = curr
                        cand_tuple = (energy, (merge_pair[0],
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
            for b, (_, (_, _, m, p)) in enumerate(beam_heap):
                parent[b] = p
                membership[b] = m

        embed()
        exit()


        return parent

    def fit(self):
        self.beam_search()


if __name__ == '__main__':
    logging.basicConfig(
            format='(trellis) :: %(asctime)s >> %(message)s',
            datefmt='%m-%d-%y %H:%M:%S',
            level=logging.INFO
    )

    with open('test_sim_graph.pkl', 'rb') as f:
        X = pickle.load(f)

    trellis = Trellis(adj_mx=X, beam_width=2)
    trellis.fit()
