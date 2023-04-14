CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var2_com_re_mask_addh_mask_ratio0.05_qm9_gap_max_z_120_finetuning --dataset-arg gap --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_var2_com_re_mask_addh_mask_ratio0.05/step=399999-epoch=8-val_loss=0.1604-test_loss=0.1501-train_per_step=0.1486.ckpt --max-z 120 > ET-PCQM4MV2_dih_var0.04_var2_com_re_mask_addh_mask_ratio0.05_qm9_gap_max_z_120_finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var2_com_re_mask_addh_mask_ratio0.05_qm9_zpve_max_z_120_finetuning --dataset-arg zpve --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_var2_com_re_mask_addh_mask_ratio0.05/step=399999-epoch=8-val_loss=0.1604-test_loss=0.1501-train_per_step=0.1486.ckpt --max-z 120 > ET-PCQM4MV2_dih_var0.04_var2_com_re_mask_addh_mask_ratio0.05_qm9_zpve_max_z_120_finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var2_com_re_mask_addh_mask_ratio0.15_qm9_gap_max_z_120_finetuning --dataset-arg gap --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_var2_com_re_mask_addh_mask_ratio0.15/step=399999-epoch=8-val_loss=0.1569-test_loss=0.1503-train_per_step=0.1538.ckpt --max-z 120 > ET-PCQM4MV2_dih_var0.04_var2_com_re_mask_addh_mask_ratio0.15_qm9_gap_max_z_120_finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var2_com_re_mask_addh_mask_ratio0.15_qm9_zpve_max_z_120_finetuning --dataset-arg zpve --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_var2_com_re_mask_addh_mask_ratio0.15/step=399999-epoch=8-val_loss=0.1569-test_loss=0.1503-train_per_step=0.1538.ckpt --max-z 120 > ET-PCQM4MV2_dih_var0.04_var2_com_re_mask_addh_mask_ratio0.15_qm9_zpve_max_z_120_finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var2_com_re_mask_addh_mask_ratio0.3_qm9_gap_max_z_120_finetuning --dataset-arg gap --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_var2_com_re_mask_addh_mask_ratio0.3/step=399999-epoch=8-val_loss=0.1590-test_loss=0.1504-train_per_step=0.1475.ckpt --max-z 120 > ET-PCQM4MV2_dih_var0.04_var2_com_re_mask_addh_mask_ratio0.3_qm9_gap_max_z_120_finetuning.log 2>&1 &


@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

CUDA_VISIBLE_DEVICES=1 python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var2_com_re_mask_addh_mask_ratio0.3_qm9_zpve_max_z_120_finetuning --dataset-arg zpve --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_var2_com_re_mask_addh_mask_ratio0.3/step=399999-epoch=8-val_loss=0.1590-test_loss=0.1504-train_per_step=0.1475.ckpt --max-z 120 > ET-PCQM4MV2_dih_var0.04_var2_com_re_mask_addh_mask_ratio0.3_qm9_zpve_max_z_120_finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var1_com_re_mask_addh_mask_ratio0.05_qm9_gap_max_z_120_finetuning --dataset-arg gap --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_var1_com_re_mask_addh_mask_ratio0.05/step=399999-epoch=8-val_loss=0.1475-test_loss=0.1431-train_per_step=0.1415.ckpt --max-z 120 > ET-PCQM4MV2_dih_var0.04_var1_com_re_mask_addh_mask_ratio0.05_qm9_gap_max_z_120_finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var1_com_re_mask_addh_mask_ratio0.05_qm9_zpve_max_z_120_finetuning --dataset-arg zpve --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_var1_com_re_mask_addh_mask_ratio0.05/step=399999-epoch=8-val_loss=0.1475-test_loss=0.1431-train_per_step=0.1415.ckpt --max-z 120 > ET-PCQM4MV2_dih_var0.04_var1_com_re_mask_addh_mask_ratio0.05_qm9_zpve_max_z_120_finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var1_com_re_mask_addh_mask_ratio0.15_qm9_gap_max_z_120_finetuning --dataset-arg gap --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_var1_com_re_mask_addh_mask_ratio0.15/step=399999-epoch=8-val_loss=0.1429-test_loss=0.1382-train_per_step=0.1463.ckpt --max-z 120 > ET-PCQM4MV2_dih_var0.04_var1_com_re_mask_addh_mask_ratio0.15_qm9_gap_max_z_120_finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var1_com_re_mask_addh_mask_ratio0.15_qm9_zpve_max_z_120_finetuning --dataset-arg zpve --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_var1_com_re_mask_addh_mask_ratio0.15/step=399999-epoch=8-val_loss=0.1429-test_loss=0.1382-train_per_step=0.1463.ckpt --max-z 120 > ET-PCQM4MV2_dih_var0.04_var1_com_re_mask_addh_mask_ratio0.15_qm9_zpve_max_z_120_finetuning.log 2>&1 &

@@@@@@@@@@@@@@@@@@


CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var1_com_re_mask_addh_mask_ratio0.3_qm9_gap_max_z_120_finetuning --dataset-arg gap --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_var1_com_re_mask_addh_mask_ratio0.3/step=399999-epoch=8-val_loss=0.1429-test_loss=0.1366-train_per_step=0.1405.ckpt --max-z 120 > ET-PCQM4MV2_dih_var0.04_var1_com_re_mask_addh_mask_ratio0.3_qm9_gap_max_z_120_finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var1_com_re_mask_addh_mask_ratio0.3_qm9_zpve_max_z_120_finetuning --dataset-arg zpve --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_var1_com_re_mask_addh_mask_ratio0.3/step=399999-epoch=8-val_loss=0.1429-test_loss=0.1366-train_per_step=0.1405.ckpt --max-z 120 > ET-PCQM4MV2_dih_var0.04_var1_com_re_mask_addh_mask_ratio0.3_qm9_zpve_max_z_120_finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var3_com_re_mask_addh_mask_ratio0.05_qm9_gap_max_z_120_finetuning --dataset-arg gap --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_var3_com_re_mask_addh_mask_ratio0.05/step=399999-epoch=8-val_loss=0.1697-test_loss=0.1566-train_per_step=0.1544.ckpt --max-z 120 > ET-PCQM4MV2_dih_var0.04_var3_com_re_mask_addh_mask_ratio0.05_qm9_gap_max_z_120_finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var3_com_re_mask_addh_mask_ratio0.05_qm9_zpve_max_z_120_finetuning --dataset-arg zpve --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_var3_com_re_mask_addh_mask_ratio0.05/step=399999-epoch=8-val_loss=0.1697-test_loss=0.1566-train_per_step=0.1544.ckpt --max-z 120 > ET-PCQM4MV2_dih_var0.04_var3_com_re_mask_addh_mask_ratio0.05_qm9_zpve_max_z_120_finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var3_com_re_mask_addh_mask_ratio0.15_qm9_gap_max_z_120_finetuning --dataset-arg gap --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_var3_com_re_mask_addh_mask_ratio0.15/step=399999-epoch=8-val_loss=0.1664-test_loss=0.1564-train_per_step=0.1581.ckpt --max-z 120 > ET-PCQM4MV2_dih_var0.04_var3_com_re_mask_addh_mask_ratio0.15_qm9_gap_max_z_120_finetuning.log 2>&1 &


@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

CUDA_VISIBLE_DEVICES=1 python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var3_com_re_mask_addh_mask_ratio0.15_qm9_zpve_max_z_120_finetuning --dataset-arg zpve --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_var3_com_re_mask_addh_mask_ratio0.15/step=399999-epoch=8-val_loss=0.1664-test_loss=0.1564-train_per_step=0.1581.ckpt --max-z 120 > ET-PCQM4MV2_dih_var0.04_var3_com_re_mask_addh_mask_ratio0.15_qm9_zpve_max_z_120_finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var3_com_re_mask_addh_mask_ratio0.3_qm9_gap_max_z_120_finetuning --dataset-arg gap --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_var3_com_re_mask_addh_mask_ratio0.3/step=399999-epoch=8-val_loss=0.1675-test_loss=0.1548-train_per_step=0.1526.ckpt --max-z 120 > ET-PCQM4MV2_dih_var0.04_var3_com_re_mask_addh_mask_ratio0.3_qm9_gap_max_z_120_finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var3_com_re_mask_addh_mask_ratio0.3_qm9_zpve_max_z_120_finetuning --dataset-arg zpve --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_var3_com_re_mask_addh_mask_ratio0.3/step=399999-epoch=8-val_loss=0.1675-test_loss=0.1548-train_per_step=0.1526.ckpt --max-z 120 > ET-PCQM4MV2_dih_var0.04_var3_com_re_mask_addh_mask_ratio0.3_qm9_zpve_max_z_120_finetuning.log 2>&1 &


