python ecc.py --data_path ../data/qian_processed.pkl --seed=0 --output_dir="../experiments/exp_ecc_qian_0_deterministic_generation" --max_sdp_iters=50000

python mlcl.py --data_path ../data/qian_processed.pkl --seed=0 --output_dir="../experiments/exp_mlcl_qian_0" --max_sdp_iters=50000
python mlcl.py --data_path ../data/qian_processed.pkl --seed=1 --output_dir="../experiments/exp_mlcl_qian_1" --max_sdp_iters=50000
python mlcl.py --data_path ../data/qian_processed.pkl --seed=2 --output_dir="../experiments/exp_mlcl_qian_2" --max_sdp_iters=50000
python mlcl.py --data_path ../data/qian_processed.pkl --seed=3 --output_dir="../experiments/exp_mlcl_qian_3" --max_sdp_iters=50000
python mlcl.py --data_path ../data/qian_processed.pkl --seed=4 --output_dir="../experiments/exp_mlcl_qian_4" --max_sdp_iters=50000
