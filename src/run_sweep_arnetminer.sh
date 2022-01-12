python ecc.py --data_path ../data/arnetminer_processed.pkl --seed=0 --output_dir="../experiments/exp_ecc_arnetminer_0_deterministic_generation" --max_sdp_iters=50000

python mlcl.py --data_path ../data/arnetminer_processed.pkl --seed=0 --output_dir="../experiments/exp_mlcl_arnetminer_0" --max_sdp_iters=50000
python mlcl.py --data_path ../data/arnetminer_processed.pkl --seed=1 --output_dir="../experiments/exp_mlcl_arnetminer_1" --max_sdp_iters=50000
python mlcl.py --data_path ../data/arnetminer_processed.pkl --seed=2 --output_dir="../experiments/exp_mlcl_arnetminer_2" --max_sdp_iters=50000
python mlcl.py --data_path ../data/arnetminer_processed.pkl --seed=3 --output_dir="../experiments/exp_mlcl_arnetminer_3" --max_sdp_iters=50000
python mlcl.py --data_path ../data/arnetminer_processed.pkl --seed=4 --output_dir="../experiments/exp_mlcl_arnetminer_4" --max_sdp_iters=50000
