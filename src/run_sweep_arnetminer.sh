python ecc.py --data_path ../data/arnetminer_processed.pkl --output_dir="../experiments/exp_ecc_arnetminer" --max_sdp_iters=50000
python mlcl.py --data_path ../data/arnetminer_processed.pkl --output_dir="../experiments/exp_mlcl_single_arnetminer" --max_sdp_iters=50000
python mlcl.py --data_path ../data/arnetminer_processed.pkl --output_dir="../experiments/exp_mlcl_batched_arnetminer" --max_sdp_iters=50000 --gen_constraints_in_batches
