python -u scripts/train.py --conf examples/ET-QM9-FT_E.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var2_com_re_qm9_electronic_spatial_extent__finetuning --dataset-arg electronic_spatial_extent --pretrained-model /share/project/sharefs-skfeng/pre-training-via-denoising/experiments/pretrain_models/ET-PCQM4MV2_dih_var0.04_var2_com_re/step=399999-epoch=8-val_loss=0.1422-test_loss=0.2784-train_per_step=0.1512.ckpt  > ET-PCQM4MV2_dih_var0.04_var2_com_re_qm9_electronic_spatial_extent__finetuning.log


python -u scripts/train.py --conf examples/ET-QM9-FT_D.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var2_com_re_qm9_dipole_moment__finetuning --dataset-arg dipole_moment --pretrained-model /share/project/sharefs-skfeng/pre-training-via-denoising/experiments/pretrain_models/ET-PCQM4MV2_dih_var0.04_var2_com_re/step=399999-epoch=8-val_loss=0.1422-test_loss=0.2784-train_per_step=0.1512.ckpt  > ET-PCQM4MV2_dih_var0.04_var2_com_re_qm9_dipole_moment__finetuning.log


python -u scripts/train.py --conf examples/ET-QM9-FT_E.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_var0.04_s_qm9_electronic_spatial_extent__finetuning --dataset-arg electronic_spatial_extent --pretrained-model /share/project/sharefs-skfeng/pre-training-via-denoising/experiments/pretrain_models/ET-PCQM4MV2_var0.04_s/step=399999-epoch=8-val_loss=0.1139-test_loss=0.2531-train_per_step=0.1102.ckpt  > ET-PCQM4MV2_var0.04_s_qm9_electronic_spatial_extent__finetuning.log

python -u scripts/train.py --conf examples/ET-QM9-FT_D.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_var0.04_s_qm9_dipole_moment__finetuning --dataset-arg dipole_moment --pretrained-model /share/project/sharefs-skfeng/pre-training-via-denoising/experiments/pretrain_models/ET-PCQM4MV2_var0.04_s/step=399999-epoch=8-val_loss=0.1139-test_loss=0.2531-train_per_step=0.1102.ckpt  > ET-PCQM4MV2_var0.04_s_qm9_dipole_moment__finetuning.log





python -u scripts/train.py --conf examples/ET-QM9-FT_E.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var2_com_re_qm9_electronic_spatial_extent_ElectronicSpatialExtent2_finetuning --dataset-arg electronic_spatial_extent --pretrained-model /share/project/sharefs-skfeng/pre-training-via-denoising/experiments/pretrain_models/ET-PCQM4MV2_dih_var0.04_var2_com_re/step=399999-epoch=8-val_loss=0.1422-test_loss=0.2784-train_per_step=0.1512.ckpt --output-model ElectronicSpatialExtent2  > ET-PCQM4MV2_dih_var0.04_var2_com_re_qm9_electronic_spatial_extent_ElectronicSpatialExtent2_finetuning.log

python -u scripts/train.py --conf examples/ET-QM9-FT_E.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_var0.04_s_qm9_electronic_spatial_extent_ElectronicSpatialExtent2_finetuning --dataset-arg electronic_spatial_extent --pretrained-model /share/project/sharefs-skfeng/pre-training-via-denoising/experiments/pretrain_models/ET-PCQM4MV2_var0.04_s/step=399999-epoch=8-val_loss=0.1139-test_loss=0.2531-train_per_step=0.1102.ckpt --output-model ElectronicSpatialExtent2 > ET-PCQM4MV2_var0.04_s_qm9_electronic_spatial_extent_ElectronicSpatialExtent2_finetuning.log


python -u scripts/train.py --conf examples/ET-QM9-FT_E.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var2_com_re_qm9_electronic_spatial_extent_VectorOutput2_finetuning --dataset-arg electronic_spatial_extent --pretrained-model /share/project/sharefs-skfeng/pre-training-via-denoising/experiments/pretrain_models/ET-PCQM4MV2_dih_var0.04_var2_com_re/step=399999-epoch=8-val_loss=0.1422-test_loss=0.2784-train_per_step=0.1512.ckpt --output-model-noise VectorOutput2  > ET-PCQM4MV2_dih_var0.04_var2_com_re_qm9_electronic_spatial_extent_VectorOutput2_finetuning.log



python -u scripts/train.py --conf examples/ET-QM9-FT_E.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var2_com_re_qm9_electronic_spatial_extent_VectorOutput2_finetuning --dataset-arg electronic_spatial_extent --pretrained-model /share/project/sharefs-skfeng/pre-training-via-denoising/experiments/pretrain_models/ET-PCQM4MV2_dih_var0.04_var2_com_re/step=399999-epoch=8-val_loss=0.1422-test_loss=0.2784-train_per_step=0.1512.ckpt --output-model-noise VectorOutput2  > ET-PCQM4MV2_dih_var0.04_var2_com_re_qm9_electronic_spatial_extent_VectorOutput2_finetuning.log