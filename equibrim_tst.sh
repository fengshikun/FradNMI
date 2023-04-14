CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_equilibrim_80epoch_addh_qm9_homo__finetuning --dataset-arg homo --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_equilibrim_80epoch_addh/step=3861039-epoch=79-val_loss=0.0988-test_loss=0.1076-train_per_step=0.0968.ckpt  > ET-PCQM4MV2_dih_var0.04_equilibrim_80epoch_addh_qm9_homo__finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_equilibrim_80epoch_addh_qm9_gap__finetuning --dataset-arg gap --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_equilibrim_80epoch_addh/step=3861039-epoch=79-val_loss=0.0988-test_loss=0.1076-train_per_step=0.0968.ckpt  > ET-PCQM4MV2_dih_var0.04_equilibrim_80epoch_addh_qm9_gap__finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=2 python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_equilibrim_80epoch_addh_qm9_lumo__finetuning --dataset-arg lumo --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_equilibrim_80epoch_addh/step=3861039-epoch=79-val_loss=0.0988-test_loss=0.1076-train_per_step=0.0968.ckpt  > ET-PCQM4MV2_dih_var0.04_equilibrim_80epoch_addh_qm9_lumo__finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_equilibrim_eqw_80epoch_addh_qm9_homo__finetuning --dataset-arg homo --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_equilibrim_eqw_80epoch_addh/step=3861039-epoch=79-val_loss=0.1144-test_loss=0.1107-train_per_step=0.1027.ckpt  > ET-PCQM4MV2_dih_var0.04_equilibrim_eqw_80epoch_addh_qm9_homo__finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_equilibrim_eqw_80epoch_addh_qm9_gap__finetuning --dataset-arg gap --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_equilibrim_eqw_80epoch_addh/step=3861039-epoch=79-val_loss=0.1144-test_loss=0.1107-train_per_step=0.1027.ckpt  > ET-PCQM4MV2_dih_var0.04_equilibrim_eqw_80epoch_addh_qm9_gap__finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=2 python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_equilibrim_eqw_80epoch_addh_qm9_lumo__finetuning --dataset-arg lumo --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_equilibrim_eqw_80epoch_addh/step=3861039-epoch=79-val_loss=0.1144-test_loss=0.1107-train_per_step=0.1027.ckpt  > ET-PCQM4MV2_dih_var0.04_equilibrim_eqw_80epoch_addh_qm9_lumo__finetuning.log 2>&1 &


