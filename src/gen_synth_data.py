from collections import namedtuple
import pickle
import random

import numpy as np
import pytorch_lightning as pl
from scipy.sparse import coo_matrix, csr_matrix

from IPython import embed


def gen_synth_data(num_clusters: int,
                   num_points: int,
                   data_dim: int,
                   cluster_feature_noise: float,
                   point_feature_sample_prob: float,
                   edge_weight_mean: float,
                   edge_weight_stddev: float):
    cluster_features, point_features, point_labels = [], [], []

    # generate the cluster features
    block_size = data_dim // num_clusters
    for cluster_idx in range(num_clusters):
        tmp_cluster = np.zeros(data_dim, dtype=int)
        tmp_cluster[cluster_idx*block_size:(cluster_idx+1)*block_size] = 1
        noise_domain = np.where(tmp_cluster == 0)[0]
        noise_mask = (np.random.uniform(0, 1, size=noise_domain.shape)
                      < cluster_feature_noise)
        noise_idx = noise_domain[noise_mask]
        tmp_cluster[noise_idx] = 1
        cluster_features.append(tmp_cluster)
    cluster_features = np.vstack(cluster_features)

    # generate the points
    for point_idx in range(num_points):
        cluster_idx = point_idx % num_clusters
        point_labels.append(cluster_idx)
        while True:
            cluster_feat_domain = np.where(
                    cluster_features[cluster_idx] == 1)[0]
            cluster_feat_mask = (
                    np.random.uniform(0, 1, size=cluster_feat_domain.shape)
                        < point_feature_sample_prob)
            cluster_feat_idx = cluster_feat_domain[cluster_feat_mask]
            sample_point = np.zeros_like(cluster_features[cluster_idx])
            sample_point[cluster_feat_idx] = 1
            subsets = np.all(
                    (sample_point & cluster_features) == sample_point,
                    axis=1)
            if subsets[cluster_idx] and np.sum(subsets) == 1:
                point_features.append(sample_point)
                break
    point_features = np.vstack(point_features)
    point_labels = np.asarray(point_labels)

    # HACK: cluster reps are exactly the aggregation of all their points
    #        (no more, no less) -> this way we don't need to add attributes
    #        which are not present in the points
    for cluster_idx in range(num_clusters):
        point_mask = (point_labels == cluster_idx)
        assert np.sum(point_mask) > 0
        cluster_features[cluster_idx] = (
                np.sum(point_features[point_mask], axis=0) > 0).astype(int)

    # assert cluster features are non-nested!!!
    assert np.all(np.argmax(cluster_features @ cluster_features.T, axis=1)
                  == np.array(range(cluster_features.shape[0])))

    # sample edge weights 
    gold_pw_labels = (point_labels[:, None] == point_labels[None, :])
    gold_pw_labels = (2.0*gold_pw_labels.astype(float)) - 1.0
    edge_weights = np.random.normal(loc=edge_weight_mean,
                                    scale=edge_weight_stddev,
                                    size=gold_pw_labels.shape)
    edge_weights = coo_matrix(np.triu(gold_pw_labels * edge_weights, k=1)).tocsr()

    dc_graph = {
            'edge_weights': edge_weights,
            'cluster_features': csr_matrix(cluster_features),
            'point_features': csr_matrix(point_features),
            'labels': point_labels
    }

    return dc_graph


if __name__ == '__main__':
    seed = 42
    pl.utilities.seed.seed_everything(seed)

    num_clusters=3
    num_points=15
    data_dim=12
    cluster_feature_noise=0.3
    point_feature_sample_prob=0.5
    edge_weight_mean=1.0
    edge_weight_stddev=2.0
    out_fname = 'tiny_data.pkl'

    tiny_data = gen_synth_data(num_clusters=num_clusters,
                               num_points=num_points,
                               data_dim=data_dim,
                               cluster_feature_noise=cluster_feature_noise,
                               point_feature_sample_prob=point_feature_sample_prob,
                               edge_weight_mean=edge_weight_mean,
                               edge_weight_stddev=edge_weight_stddev)

    with open(out_fname, 'wb') as f:
        pickle.dump(tiny_data, f)
