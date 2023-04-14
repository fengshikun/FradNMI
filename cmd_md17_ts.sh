python -u scripts/train.py --conf examples/ET-MD17_9500.yaml  --job-id train_from_scratch_md17_malonaldehyde_finetuning --dataset-arg malonaldehyde > train_from_scratch_qm9_malonaldehyde_finetuning.log
python -u scripts/train.py --conf examples/ET-MD17_9500.yaml  --job-id train_from_scratch_md17_aspirin_finetuning --dataset-arg aspirin > train_from_scratch_qm9_aspirin_finetuning.log
python -u scripts/train.py --conf examples/ET-MD17_9500.yaml  --job-id train_from_scratch_md17_ethanol_finetuning --dataset-arg ethanol > train_from_scratch_qm9_ethanol_finetuning.log
python -u scripts/train.py --conf examples/ET-MD17_9500.yaml  --job-id train_from_scratch_md17_salicylic_acid_finetuning --dataset-arg salicylic_acid > train_from_scratch_qm9_salicylic_acid_finetuning.log
python -u scripts/train.py --conf examples/ET-MD17_9500.yaml  --job-id train_from_scratch_md17_benzene_finetuning --dataset-arg benzene > train_from_scratch_qm9_benzene_finetuning.log


python -u scripts/train.py --conf examples/ET-MD17_9500.yaml  --job-id train_from_scratch_md17_naphthalene_finetuning --dataset-arg naphthalene > train_from_scratch_qm9_naphthalene_finetuning.log
python -u scripts/train.py --conf examples/ET-MD17_9500.yaml  --job-id train_from_scratch_md17_uracil_finetuning --dataset-arg uracil > train_from_scratch_qm9_uracil_finetuning.log
python -u scripts/train.py --conf examples/ET-MD17_9500.yaml  --job-id train_from_scratch_md17_toluene_finetuning --dataset-arg toluene > train_from_scratch_qm9_toluene_finetuning.log
