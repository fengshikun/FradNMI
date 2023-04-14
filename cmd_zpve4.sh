CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.1_long.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var2_com_re_long_qm9_zpve__finetuning --dataset-arg zpve --pretrained-model /share/project/sharefs-skfeng/pre-training-via-denoising/experiments/pretrain_models/ET-PCQM4MV2_dih_var0.04_var2_com_re/step=399999-epoch=8-val_loss=0.1422-test_loss=0.2784-train_per_step=0.1512.ckpt  > ET-PCQM4MV2_dih_var0.04_var2_com_re_long_qm9_zpve__finetuning.log 2>&1 &



python -u scripts/train.py --conf examples/ET-QM9-FT_E.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var2_com_re_mask_addh_vector_output2_qm9_electronic_spatial_extent__finetuning --dataset-arg electronic_spatial_extent --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_var2_com_re_mask_addh_vector_output2/step=399999-epoch=8-val_loss=0.1544-test_loss=0.1475-train_per_step=0.1440.ckpt  > ET-PCQM4MV2_dih_var0.04_var2_com_re_mask_addh_vector_output2_qm9_electronic_spatial_extent__finetuning.log
