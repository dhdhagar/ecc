from collections import defaultdict
import glob
import os
import pickle
import re

import numpy as np

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
                         gold_labels,
                         num_clusters_by_block,
                         pred_clusterings,
                         cu_by_block,
                         fmc_by_block):
    pass


if __name__ == '__main__':

    expt_dirs = glob.glob('../experiments/*')

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

        single_expt_df = 
        
        embed()
        exit()
