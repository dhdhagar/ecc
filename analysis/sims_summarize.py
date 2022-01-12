from collections import defaultdict
import glob
import os
import pickle
import re

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score as rand_idx
from sklearn.metrics import homogeneity_completeness_v_measure as cluster_f1

from IPython import embed


def extract_info_from_log(log_fname):

    constraint_type = None
    cu_by_block = defaultdict(list)
    fmc_by_block = defaultdict(list)
    current_block = None

    with open(log_fname, 'r') as f:
        for line in f:
            line.strip()

            # extract constraint type
            if constraint_type is None:
                re_constraint_type = re.search('^\(([A-Z]+)\).*$', line)
                if re_constraint_type:
                    constraint_type = re_constraint_type.group(1)

            # extract current block name
            re_start_new_block = re.search('Loaded block "(.*)" .*$', line)
            if re_start_new_block:
                current_block = re_start_new_block.group(1)

            # extract round info
            if constraint_type == 'MLCL':
                re_round_info = re.search('num_mlcl = (\d+);.* match_feat_coeff = ([\.0-9]+);', line)
                if re_round_info:
                    cu_by_block[current_block].append(2*int(re_round_info.group(1)))
                    fmc_by_block[current_block].append(float(re_round_info.group(2)))
            elif constraint_type == 'ECC':
                re_round_info = re.search('num_ecc_feats = (\d+);.* match_feat_coeff = ([\.0-9]+);', line)
                if re_round_info:
                    cu_by_block[current_block].append(int(re_round_info.group(1)))
                    fmc_by_block[current_block].append(float(re_round_info.group(2)))

    return constraint_type, dict(cu_by_block), dict(fmc_by_block)


def build_single_expt_df(constraint_type,
                         seed,
                         gold_labels,
                         num_clusters_by_block,
                         pred_clusterings,
                         cu_by_block,
                         fmc_by_block):

    block_names = sorted(list(gold_labels.keys()),
                         key=lambda x: -gold_labels[x].size)

    block_start, block_end = {}, {}
    block_curr_fmc = {}

    num_points_so_far = 0
    gold_cluster_labels = []
    pred_cluster_labels = []
    for block_name in block_names:
        block_start[block_name] = num_points_so_far
        gold_cluster_labels.append(gold_labels[block_name] + num_points_so_far)
        pred_cluster_labels.append(pred_clusterings[block_name][0] + num_points_so_far)
        num_points_so_far += gold_labels[block_name].size
        block_end[block_name] = num_points_so_far
        block_curr_fmc[block_name] = fmc_by_block[block_name][0]
    gold_cluster_labels = np.concatenate(gold_cluster_labels, axis=0)
    pred_cluster_labels = np.concatenate(pred_cluster_labels, axis=0)

    df = pd.DataFrame(
            columns=(
                'constraint_type',
                'seed',
                '# constraints',
                'CU',
                'FMC',
                'Rand Idx',
                'F1'
            )
    )

    curr_cu = 0
    curr_fmc = sum(
            [block_curr_fmc[block_name] * num_clusters_by_block[block_name]
                for block_name in block_names]
    )
    curr_fmc /= sum(num_clusters_by_block.values())
    curr_rand_idx = rand_idx(gold_cluster_labels, pred_cluster_labels)
    curr_f1 = cluster_f1(gold_cluster_labels, pred_cluster_labels)[2]

    df.loc[0] = [
            constraint_type,
            seed,
            0,
            curr_cu,
            curr_fmc,
            curr_rand_idx,
            curr_f1
    ]

    row_num = 1
    for i in range(1, max([len(x) for x in cu_by_block.values()])):
        for block_name in block_names:
            if i >= len(fmc_by_block[block_name]):
                continue

            block_curr_fmc[block_name] = fmc_by_block[block_name][i]
            
            start = block_start[block_name]
            end = block_end[block_name]
            pred_cluster_labels[start:end] = pred_clusterings[block_name][i] + start

            curr_cu += (cu_by_block[block_name][i] - cu_by_block[block_name][i-1])
            curr_fmc = sum(
                    [block_curr_fmc[block_name] * num_clusters_by_block[block_name]
                        for block_name in block_names]
            )
            curr_fmc /= sum(num_clusters_by_block.values())
            curr_rand_idx = rand_idx(gold_cluster_labels, pred_cluster_labels)
            curr_f1 = cluster_f1(gold_cluster_labels, pred_cluster_labels)[2]

            df.loc[row_num] = [
                    constraint_type,
                    seed,
                    row_num,
                    curr_cu,
                    curr_fmc,
                    curr_rand_idx,
                    curr_f1
            ]

            row_num += 1

    return df


if __name__ == '__main__':

    expt_dirs = glob.glob('../experiments/*mlcl_*_[0-4]')
    expt_dirs.extend(glob.glob('../experiments/*ecc_*deterministic*'))

    expt_result_dfs = []
    for expt_dir in expt_dirs:
        (constraint_type, cu_by_block, fmc_by_block) = extract_info_from_log(
                os.path.join(expt_dir, 'out.log'))
        hparams = pickle.load(open(os.path.join(expt_dir, 'hparams.pkl'), 'rb'))
        with open(hparams.data_path, 'rb') as f:
            input_data = pickle.load(f)
            gold_labels = {k: v['labels'] for k, v in input_data.items()}
            num_clusters_by_block = {k: np.unique(v['labels']).size for k, v in input_data.items()}

        pred_clusterings = pickle.load(
                open(os.path.join(expt_dir, 'pred_clusterings.pkl'), 'rb'))

        single_expt_df = build_single_expt_df(
                constraint_type,
                hparams.seed,
                gold_labels,
                num_clusters_by_block,
                pred_clusterings,
                cu_by_block,
                fmc_by_block
        )

        expt_result_dfs.append(single_expt_df)

    # create final endpoints for nice plotting
    agg_df = pd.concat(expt_result_dfs, ignore_index=True)
    max_num_constraints = agg_df['# constraints'].max()
    max_cu = agg_df['CU'].max()
    for df in expt_result_dfs:
        constraint_type = df['constraint_type'][0]
        seed = df['seed'][0]
        df.loc[len(df)] = [
                constraint_type,
                seed,
                max_num_constraints,
                max_cu,
                1.0,
                1.0,
                1.0
        ]

    # dump aggregated df to pickle file
    with open('sims_summary_df.pkl', 'wb') as f:
        pickle.dump(pd.concat(expt_result_dfs, ignore_index=True), f)
