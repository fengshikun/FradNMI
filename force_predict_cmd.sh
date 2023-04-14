# python scripts/train.py --conf examples/ET-PCQM4V2_dih_var0.04_var2_predict_force.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_1w_force_field --force_field true  > ET-PCQM4MV2_dih_var0.04_1w_force_field.log;
# python scripts/train.py --conf examples/ET-PCQM4V2_dih_var0.04_var2_predict_force.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_1w_predict_noise --pred_noise true  > ET-PCQM4MV2_dih_var0.04_1w_predict_noise.log;
# python scripts/train.py --conf examples/ET-PCQM4V2_dih_var0.04_var2_predict_force.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_1w_frad  > ET-PCQM4MV2_dih_var0.04_1w_frad.log;

# python scripts/train.py --conf examples/ET-PCQM4V2_dih_var0.04_var2_predict_force.yaml --layernorm-on-vec whitened --job-id pretraining_1w_guassian_noise_8epoch --cod_denoise true --num-epochs 8  > pretraining_1w_guassian_noise_8epoch.log;


# python scripts/train.py --conf examples/ET-PCQM4V2_dih_var0.04_var2_predict_force.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_1w_predict_noise_8epoch --pred_noise true --num-epochs 8  > ET-PCQM4MV2_dih_var0.04_1w_predict_noise_8epoch.log;
# python scripts/train.py --conf examples/ET-PCQM4V2_dih_var0.04_var2_predict_force.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_1w_frad_8epoch --num-epochs 8  > ET-PCQM4MV2_dih_var0.04_1w_frad_8epoch.log;


# python scripts/train.py --conf examples/ET-PCQM4V2_dih_var0.04_var2_predict_force.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_1w_force_field_8epoch --force_field true  --num-epochs 8 > ET-PCQM4MV2_dih_var0.04_1w_force_field_8epoch.log;

# /home/fengshikun/Pretraining-Denoising/examples/ET-PCQM4V2_dih_var0.04_var2_predict_force_md17.yaml

# python scripts/train.py --conf examples/ET-PCQM4V2_dih_var0.04_var2_predict_force_md17.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_1w_force_field_md17 --force_field true  > ET-PCQM4MV2_dih_var0.04_1w_force_field_md17.log;


# python scripts/train.py --conf examples/ET-PCQM4V2_dih_var0.04_var2_predict_force_md17.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_1w_frad_md17  > ET-PCQM4MV2_dih_var0.04_1w_frad_md17.log;


# python scripts/train.py --conf examples/ET-PCQM4V2_dih_var0.04_var2_predict_force_md17.yaml --layernorm-on-vec whitened --job-id pretraining_1w_guassian_noise_8epoch_md17 --cod_denoise true --num-epochs 8  > pretraining_1w_guassian_noise_8epoch_md17.log


# python scripts/train.py --conf examples/ET-PCQM4V2_dih_var0.04_var2_predict_force_md17.yaml --layernorm-on-vec whitened --job-id pretraining_rdkit_guassian_noise --cod_denoise true --rdkit_conf true > pretraining_rdkit_guassian_noise.log

# python scripts/train.py --conf examples/ET-PCQM4V2_dih_var0.04_var2_predict_force_md17.yaml --layernorm-on-vec whitened --job-id pretraining_rdkit_frad --rdkit_conf true > pretraining_rdkit_frad.log

python scripts/train.py --conf examples/ET-PCQM4V2_dih_var0.04_var2_predict_force.yaml --layernorm-on-vec whitened --job-id pretraining_rdkit_guassian_noise_qm9 --cod_denoise true --rdkit_conf true > pretraining_rdkit_guassian_noise_qm9.log

python scripts/train.py --conf examples/ET-PCQM4V2_dih_var0.04_var2_predict_force.yaml --layernorm-on-vec whitened --job-id pretraining_rdkit_frad_qm9 --rdkit_conf true > pretraining_rdkit_frad_qm9.log