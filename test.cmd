python -u scripts/train.py --conf examples/ET-QM9-FT-nt_lr_0.001.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var2_com_re_qm9_energy_U0_finetuning_lr_0.001 --dataset-arg energy_U0 --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_var2_com_re/step=399999-epoch=8-val_loss=0.1422-test_loss=0.2784-train_per_step=0.1512.ckpt > ET-PCQM4MV2_dih_var0.04_var2_com_re_qm9_energy_U0_finetuning_lr_0.001.log 





python -u scripts/train.py --conf examples/ET-QM9-FT-nt_lr_0.001.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var2_com_re_qm9_energy_U_finetuning --dataset-arg energy_U --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_var2_com_re/step=399999-epoch=8-val_loss=0.1422-test_loss=0.2784-train_per_step=0.1512.ckpt > ET-PCQM4MV2_dih_var0.04_var2_com_re_qm9_energy_U_finetuning.log 2>&1 &
python -u scripts/train.py --conf examples/ET-QM9-FT-nt_lr_0.001.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var2_com_re_qm9_enthalpy_H_finetuning --dataset-arg enthalpy_H --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_var2_com_re/step=399999-epoch=8-val_loss=0.1422-test_loss=0.2784-train_per_step=0.1512.ckpt > ET-PCQM4MV2_dih_var0.04_var2_com_re_qm9_enthalpy_H_finetuning.log 2>&1 &
python -u scripts/train.py --conf examples/ET-QM9-FT-nt_lr_0.001.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var2_com_re_qm9_free_energy_finetuning --dataset-arg free_energy --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_var2_com_re/step=399999-epoch=8-val_loss=0.1422-test_loss=0.2784-train_per_step=0.1512.ckpt > ET-PCQM4MV2_dih_var0.04_var2_com_re_qm9_free_energy_finetuning.log 2>&1 &