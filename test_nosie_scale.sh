CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.4_var2_com_re_qm9_homo__finetuning --dataset-arg homo --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.4_var2_com_re/step=399999-epoch=8-val_loss=0.1705-test_loss=0.1385-train_per_step=0.1500.ckpt  > ET-PCQM4MV2_dih_var0.4_var2_com_re_qm9_homo__finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.4_var2_com_re_qm9_gap__finetuning --dataset-arg gap --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.4_var2_com_re/step=399999-epoch=8-val_loss=0.1705-test_loss=0.1385-train_per_step=0.1500.ckpt  > ET-PCQM4MV2_dih_var0.4_var2_com_re_qm9_gap__finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=2 python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.4_var2_com_re_qm9_lumo__finetuning --dataset-arg lumo --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.4_var2_com_re/step=399999-epoch=8-val_loss=0.1705-test_loss=0.1385-train_per_step=0.1500.ckpt  > ET-PCQM4MV2_dih_var0.4_var2_com_re_qm9_lumo__finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=3 python -u scripts/train.py --conf examples/ET-QM9-FT-nt.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.4_var2_com_re_qm9_energy_U0__finetuning --dataset-arg energy_U0 --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.4_var2_com_re/step=399999-epoch=8-val_loss=0.1705-test_loss=0.1385-train_per_step=0.1500.ckpt  > ET-PCQM4MV2_dih_var0.4_var2_com_re_qm9_energy_U0__finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=4 python -u scripts/train.py --conf examples/ET-QM9-FT-nt.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.4_var2_com_re_qm9_energy_U__finetuning --dataset-arg energy_U --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.4_var2_com_re/step=399999-epoch=8-val_loss=0.1705-test_loss=0.1385-train_per_step=0.1500.ckpt  > ET-PCQM4MV2_dih_var0.4_var2_com_re_qm9_energy_U__finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=5 python -u scripts/train.py --conf examples/ET-QM9-FT-nt.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.4_var2_com_re_qm9_enthalpy_H__finetuning --dataset-arg enthalpy_H --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.4_var2_com_re/step=399999-epoch=8-val_loss=0.1705-test_loss=0.1385-train_per_step=0.1500.ckpt  > ET-PCQM4MV2_dih_var0.4_var2_com_re_qm9_enthalpy_H__finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=6 python -u scripts/train.py --conf examples/ET-QM9-FT-nt.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.4_var2_com_re_qm9_free_energy__finetuning --dataset-arg free_energy --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.4_var2_com_re/step=399999-epoch=8-val_loss=0.1705-test_loss=0.1385-train_per_step=0.1500.ckpt  > ET-PCQM4MV2_dih_var0.4_var2_com_re_qm9_free_energy__finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.004_var2_com_re_qm9_homo__finetuning --dataset-arg homo --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.004_var2_com_re/step=399999-epoch=8-val_loss=0.2646-test_loss=0.2472-train_per_step=0.2534.ckpt  > ET-PCQM4MV2_dih_var0.004_var2_com_re_qm9_homo__finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.004_var2_com_re_qm9_gap__finetuning --dataset-arg gap --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.004_var2_com_re/step=399999-epoch=8-val_loss=0.2646-test_loss=0.2472-train_per_step=0.2534.ckpt  > ET-PCQM4MV2_dih_var0.004_var2_com_re_qm9_gap__finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=2 python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.004_var2_com_re_qm9_lumo__finetuning --dataset-arg lumo --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.004_var2_com_re/step=399999-epoch=8-val_loss=0.2646-test_loss=0.2472-train_per_step=0.2534.ckpt  > ET-PCQM4MV2_dih_var0.004_var2_com_re_qm9_lumo__finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=3 python -u scripts/train.py --conf examples/ET-QM9-FT-nt.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.004_var2_com_re_qm9_energy_U0__finetuning --dataset-arg energy_U0 --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.004_var2_com_re/step=399999-epoch=8-val_loss=0.2646-test_loss=0.2472-train_per_step=0.2534.ckpt  > ET-PCQM4MV2_dih_var0.004_var2_com_re_qm9_energy_U0__finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=4 python -u scripts/train.py --conf examples/ET-QM9-FT-nt.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.004_var2_com_re_qm9_energy_U__finetuning --dataset-arg energy_U --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.004_var2_com_re/step=399999-epoch=8-val_loss=0.2646-test_loss=0.2472-train_per_step=0.2534.ckpt  > ET-PCQM4MV2_dih_var0.004_var2_com_re_qm9_energy_U__finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=5 python -u scripts/train.py --conf examples/ET-QM9-FT-nt.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.004_var2_com_re_qm9_enthalpy_H__finetuning --dataset-arg enthalpy_H --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.004_var2_com_re/step=399999-epoch=8-val_loss=0.2646-test_loss=0.2472-train_per_step=0.2534.ckpt  > ET-PCQM4MV2_dih_var0.004_var2_com_re_qm9_enthalpy_H__finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=6 python -u scripts/train.py --conf examples/ET-QM9-FT-nt.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.004_var2_com_re_qm9_free_energy__finetuning --dataset-arg free_energy --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.004_var2_com_re/step=399999-epoch=8-val_loss=0.2646-test_loss=0.2472-train_per_step=0.2534.ckpt  > ET-PCQM4MV2_dih_var0.004_var2_com_re_qm9_free_energy__finetuning.log 2>&1 &


