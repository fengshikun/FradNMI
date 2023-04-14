python -u scripts/train.py --conf examples/ET-QM9-FT-nt.yaml --layernorm-on-vec whitened --job-id pretraining_1w_guassian_noise_8epoch_qm9_energy_U0_denoising_weight_0_finetuning --dataset-arg energy_U0 --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/pretraining_1w_guassian_noise_8epoch/step=1119-epoch=7-val_loss=0.4481-test_loss=0.4140-train_per_step=0.4146.ckpt --denoising-weight 0 > pretraining_1w_guassian_noise_8epoch_qm9_energy_U0_denoising_weight_0_finetuning.log

python -u scripts/train.py --conf examples/ET-QM9-FT-nt.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_1w_frad_qm9_energy_U0_denoising_weight_0_finetuning --dataset-arg energy_U0 --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_1w_frad/step=4199-epoch=29-val_loss=0.3122-test_loss=0.3035-train_per_step=0.2895.ckpt --denoising-weight 0 > ET-PCQM4MV2_dih_var0.04_1w_frad_qm9_energy_U0_denoising_weight_0_finetuning.log


python -u scripts/train.py --conf examples/ET-QM9-FT-nt.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_1w_force_field_qm9_energy_U0_denoising_weight_0_finetuning --dataset-arg energy_U0 --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_1w_force_field/step=4199-epoch=29-val_loss=0.0204-test_loss=0.0201-train_per_step=0.0165.ckpt --denoising-weight 0 > ET-PCQM4MV2_dih_var0.04_1w_force_field_qm9_energy_U0_denoising_weight_0_finetuning.log


python -u scripts/train.py --conf examples/ET-QM9-FT-nt.yaml --layernorm-on-vec whitened --job-id train_from_scratch_qm9_energy_U0_denoising_weight_0_finetuning --dataset-arg energy_U0 --denoising-weight 0 > train_from_scratch_qm9_energy_U0_denoising_weight_0_finetuning.log


