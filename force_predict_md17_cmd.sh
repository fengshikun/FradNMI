CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py --conf examples/ET-MD17.yaml  --job-id pretraining_1w_guassian_noise_8epoch_md17_md17_aspirin__finetuning --dataset-arg aspirin --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/pretraining_1w_guassian_noise_8epoch_md17/step=1119-epoch=7-val_loss=0.5406-test_loss=0.4976-train_per_step=0.4958.ckpt  > pretraining_1w_guassian_noise_8epoch_md17_qm9_aspirin__finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 python -u scripts/train.py --conf examples/ET-MD17.yaml  --job-id ET-PCQM4MV2_dih_var0.04_1w_frad_md17_md17_aspirin__finetuning --dataset-arg aspirin --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_1w_frad_md17/step=4199-epoch=29-val_loss=0.3514-test_loss=0.3528-train_per_step=0.3307.ckpt  > ET-PCQM4MV2_dih_var0.04_1w_frad_md17_qm9_aspirin__finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=2 python -u scripts/train.py --conf examples/ET-MD17.yaml  --job-id ET-PCQM4MV2_dih_var0.04_1w_force_field_md17_md17_aspirin__finetuning --dataset-arg aspirin --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_1w_force_field_md17/step=4199-epoch=29-val_loss=0.0263-test_loss=0.0261-train_per_step=0.0264.ckpt  > ET-PCQM4MV2_dih_var0.04_1w_force_field_md17_qm9_aspirin__finetuning.log 2>&1 &


