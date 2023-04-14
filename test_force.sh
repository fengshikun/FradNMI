python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id train_from_scratch_qm9_homo_finetuning2 --dataset-arg homo > train_from_scratch_qm9_homo__finetuning2.log 


python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id train_from_scratch_qm9_gap_finetuning2 --dataset-arg gap   > train_from_scratch_qm9_gap_finetuning2.log 


python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id train_from_scratch_qm9_lumo_finetuning2 --dataset-arg lumo  > train_from_scratch_qm9_lumo_finetuning2.log 




