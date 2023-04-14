CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var2_com_re_equilibrim_80epoch_qm9_homo__finetuning --dataset-arg homo --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_var2_com_re_equilibrim_80epoch/step=3812776-epoch=78-val_loss=0.1101-test_loss=0.1063-train_per_step=0.1177.ckpt  > ET-PCQM4MV2_dih_var0.04_var2_com_re_equilibrim_80epoch_qm9_homo__finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var2_com_re_equilibrim_80epoch_qm9_gap__finetuning --dataset-arg gap --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_var2_com_re_equilibrim_80epoch/step=3812776-epoch=78-val_loss=0.1101-test_loss=0.1063-train_per_step=0.1177.ckpt  > ET-PCQM4MV2_dih_var0.04_var2_com_re_equilibrim_80epoch_qm9_gap__finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=2 python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var2_com_re_equilibrim_80epoch_qm9_lumo__finetuning --dataset-arg lumo --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_var2_com_re_equilibrim_80epoch/step=3812776-epoch=78-val_loss=0.1101-test_loss=0.1063-train_per_step=0.1177.ckpt  > ET-PCQM4MV2_dih_var0.04_var2_com_re_equilibrim_80epoch_qm9_lumo__finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=3 python -u scripts/train.py --conf examples/ET-QM9-FT-nt.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var2_com_re_equilibrim_80epoch_qm9_energy_U0__finetuning --dataset-arg energy_U0 --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_var2_com_re_equilibrim_80epoch/step=3812776-epoch=78-val_loss=0.1101-test_loss=0.1063-train_per_step=0.1177.ckpt  > ET-PCQM4MV2_dih_var0.04_var2_com_re_equilibrim_80epoch_qm9_energy_U0__finetuning.log 2>&1 &



CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var2_com_re_equilibrim_eqw_80epoch_qm9_homo__finetuning --dataset-arg homo --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_var2_com_re_equilibrim_eqw_80epoch/step=3812776-epoch=78-val_loss=0.1160-test_loss=0.1166-train_per_step=0.1267.ckpt  > ET-PCQM4MV2_dih_var0.04_var2_com_re_equilibrim_eqw_80epoch_qm9_homo__finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var2_com_re_equilibrim_eqw_80epoch_qm9_gap__finetuning --dataset-arg gap --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_var2_com_re_equilibrim_eqw_80epoch/step=3812776-epoch=78-val_loss=0.1160-test_loss=0.1166-train_per_step=0.1267.ckpt  > ET-PCQM4MV2_dih_var0.04_var2_com_re_equilibrim_eqw_80epoch_qm9_gap__finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=2 python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var2_com_re_equilibrim_eqw_80epoch_qm9_lumo__finetuning --dataset-arg lumo --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_var2_com_re_equilibrim_eqw_80epoch/step=3812776-epoch=78-val_loss=0.1160-test_loss=0.1166-train_per_step=0.1267.ckpt  > ET-PCQM4MV2_dih_var0.04_var2_com_re_equilibrim_eqw_80epoch_qm9_lumo__finetuning.log 2>&1 &



CUDA_VISIBLE_DEVICES=3 python -u scripts/train.py --conf examples/ET-QM9-FT-nt.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var2_com_re_equilibrim_eqw_80epoch_qm9_energy_U0__finetuning --dataset-arg energy_U0 --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_var2_com_re_equilibrim_eqw_80epoch/step=3812776-epoch=78-val_loss=0.1160-test_loss=0.1166-train_per_step=0.1267.ckpt  > ET-PCQM4MV2_dih_var0.04_var2_com_re_equilibrim_eqw_80epoch_qm9_energy_U0__finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var3_com_re_40epoch_qm9_homo__finetuning --dataset-arg homo --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_var3_com_re_40epoch/step=1930519-epoch=39-val_loss=0.1354-test_loss=0.2757-train_per_step=0.1319.ckpt  > ET-PCQM4MV2_dih_var0.04_var3_com_re_40epoch_qm9_homo__finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var3_com_re_40epoch_qm9_gap__finetuning --dataset-arg gap --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_var3_com_re_40epoch/step=1930519-epoch=39-val_loss=0.1354-test_loss=0.2757-train_per_step=0.1319.ckpt  > ET-PCQM4MV2_dih_var0.04_var3_com_re_40epoch_qm9_gap__finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=2 python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var3_com_re_40epoch_qm9_lumo__finetuning --dataset-arg lumo --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_var3_com_re_40epoch/step=1930519-epoch=39-val_loss=0.1354-test_loss=0.2757-train_per_step=0.1319.ckpt  > ET-PCQM4MV2_dih_var0.04_var3_com_re_40epoch_qm9_lumo__finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=3 python -u scripts/train.py --conf examples/ET-QM9-FT-nt.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var3_com_re_40epoch_qm9_energy_U0__finetuning --dataset-arg energy_U0 --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_var3_com_re_40epoch/step=1930519-epoch=39-val_loss=0.1354-test_loss=0.2757-train_per_step=0.1319.ckpt  > ET-PCQM4MV2_dih_var0.04_var3_com_re_40epoch_qm9_energy_U0__finetuning.log 2>&1 &


+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var4_com_re_40epoch_qm9_homo__finetuning --dataset-arg homo --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_var4_com_re_40epoch/step=1930519-epoch=39-val_loss=0.1383-test_loss=0.2784-train_per_step=0.1357.ckpt  > ET-PCQM4MV2_dih_var0.04_var4_com_re_40epoch_qm9_homo__finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var4_com_re_40epoch_qm9_gap__finetuning --dataset-arg gap --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_var4_com_re_40epoch/step=1930519-epoch=39-val_loss=0.1383-test_loss=0.2784-train_per_step=0.1357.ckpt  > ET-PCQM4MV2_dih_var0.04_var4_com_re_40epoch_qm9_gap__finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=2 python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var4_com_re_40epoch_qm9_lumo__finetuning --dataset-arg lumo --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_var4_com_re_40epoch/step=1930519-epoch=39-val_loss=0.1383-test_loss=0.2784-train_per_step=0.1357.ckpt  > ET-PCQM4MV2_dih_var0.04_var4_com_re_40epoch_qm9_lumo__finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=3 python -u scripts/train.py --conf examples/ET-QM9-FT-nt.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var4_com_re_40epoch_qm9_energy_U0__finetuning --dataset-arg energy_U0 --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_var4_com_re_40epoch/step=1930519-epoch=39-val_loss=0.1383-test_loss=0.2784-train_per_step=0.1357.ckpt  > ET-PCQM4MV2_dih_var0.04_var4_com_re_40epoch_qm9_energy_U0__finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var1_com_re_40epoch_qm9_homo__finetuning --dataset-arg homo --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_var1_com_re_40epoch/step=1930519-epoch=39-val_loss=0.1228-test_loss=0.2664-train_per_step=0.1209.ckpt  > ET-PCQM4MV2_dih_var0.04_var1_com_re_40epoch_qm9_homo__finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var1_com_re_40epoch_qm9_gap__finetuning --dataset-arg gap --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_var1_com_re_40epoch/step=1930519-epoch=39-val_loss=0.1228-test_loss=0.2664-train_per_step=0.1209.ckpt  > ET-PCQM4MV2_dih_var0.04_var1_com_re_40epoch_qm9_gap__finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=2 python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var1_com_re_40epoch_qm9_lumo__finetuning --dataset-arg lumo --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_var1_com_re_40epoch/step=1930519-epoch=39-val_loss=0.1228-test_loss=0.2664-train_per_step=0.1209.ckpt  > ET-PCQM4MV2_dih_var0.04_var1_com_re_40epoch_qm9_lumo__finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=3 python -u scripts/train.py --conf examples/ET-QM9-FT-nt.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var1_com_re_40epoch_qm9_energy_U0__finetuning --dataset-arg energy_U0 --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_var1_com_re_40epoch/step=1930519-epoch=39-val_loss=0.1228-test_loss=0.2664-train_per_step=0.1209.ckpt  > ET-PCQM4MV2_dih_var0.04_var1_com_re_40epoch_qm9_energy_U0__finetuning.log 2>&1 &


