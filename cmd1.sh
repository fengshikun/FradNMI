CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var3_com_re_mask_addh_mask_ratio0.75_qm9_gap_max_z_120_finetuning --dataset-arg gap --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_var3_com_re_mask_addh_mask_ratio0.75/step=399999-epoch=8-val_loss=0.1669-test_loss=0.1558-train_per_step=0.1578.ckpt --max-z 120 > ET-PCQM4MV2_dih_var0.04_var3_com_re_mask_addh_mask_ratio0.75_qm9_gap_max_z_120_finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var3_com_re_mask_addh_mask_ratio0.75_qm9_zpve_max_z_120_finetuning --dataset-arg zpve --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_var3_com_re_mask_addh_mask_ratio0.75/step=399999-epoch=8-val_loss=0.1669-test_loss=0.1558-train_per_step=0.1578.ckpt --max-z 120 > ET-PCQM4MV2_dih_var0.04_var3_com_re_mask_addh_mask_ratio0.75_qm9_zpve_max_z_120_finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var2_com_re_mask_addh_mask_ratio0.75_qm9_gap_max_z_120_finetuning --dataset-arg gap --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_var2_com_re_mask_addh_mask_ratio0.75/step=399999-epoch=8-val_loss=0.1587-test_loss=0.1500-train_per_step=0.1532.ckpt --max-z 120 > ET-PCQM4MV2_dih_var0.04_var2_com_re_mask_addh_mask_ratio0.75_qm9_gap_max_z_120_finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var2_com_re_mask_addh_mask_ratio0.75_qm9_zpve_max_z_120_finetuning --dataset-arg zpve --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_var2_com_re_mask_addh_mask_ratio0.75/step=399999-epoch=8-val_loss=0.1587-test_loss=0.1500-train_per_step=0.1532.ckpt --max-z 120 > ET-PCQM4MV2_dih_var0.04_var2_com_re_mask_addh_mask_ratio0.75_qm9_zpve_max_z_120_finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var1_com_re_mask_addh_mask_ratio0.75_qm9_gap_max_z_120_finetuning --dataset-arg gap --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_var1_com_re_mask_addh_mask_ratio0.75/step=399999-epoch=8-val_loss=0.1450-test_loss=0.1386-train_per_step=0.1448.ckpt --max-z 120 > ET-PCQM4MV2_dih_var0.04_var1_com_re_mask_addh_mask_ratio0.75_qm9_gap_max_z_120_finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var1_com_re_mask_addh_mask_ratio0.75_qm9_zpve_max_z_120_finetuning --dataset-arg zpve --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_var1_com_re_mask_addh_mask_ratio0.75/step=399999-epoch=8-val_loss=0.1450-test_loss=0.1386-train_per_step=0.1448.ckpt --max-z 120 > ET-PCQM4MV2_dih_var0.04_var1_com_re_mask_addh_mask_ratio0.75_qm9_zpve_max_z_120_finetuning.log 2>&1 &


