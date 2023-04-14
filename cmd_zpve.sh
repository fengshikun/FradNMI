python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var2_com_re_qm9_zpve_train_loss_type_smooth_l1_loss_finetuning --dataset-arg zpve --pretrained-model /share/project/sharefs-skfeng/pre-training-via-denoising/experiments/pretrain_models/ET-PCQM4MV2_dih_var0.04_var2_com_re/step=399999-epoch=8-val_loss=0.1422-test_loss=0.2784-train_per_step=0.1512.ckpt --train-loss-type smooth_l1_loss > ET-PCQM4MV2_dih_var0.04_var2_com_re_qm9_zpve_train_loss_type_smooth_l1_loss_finetuning.log


python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.1_long.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var2_com_re_long_dw0.1_qm9_zpve_train_loss_type_smooth_l1_loss_finetuning --dataset-arg zpve --pretrained-model /share/project/sharefs-skfeng/pre-training-via-denoising/experiments/pretrain_models/ET-PCQM4MV2_dih_var0.04_var2_com_re/step=399999-epoch=8-val_loss=0.1422-test_loss=0.2784-train_per_step=0.1512.ckpt --train-loss-type smooth_l1_loss > ET-PCQM4MV2_dih_var0.04_var2_com_re_long_dw0.1_qm9_zpve_train_loss_type_smooth_l1_loss_finetuning.log


python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var2_com_re_long_dw0.2_qm9_zpve_train_loss_type_smooth_l1_loss_finetuning --dataset-arg zpve --pretrained-model /share/project/sharefs-skfeng/pre-training-via-denoising/experiments/pretrain_models/ET-PCQM4MV2_dih_var0.04_var2_com_re/step=399999-epoch=8-val_loss=0.1422-test_loss=0.2784-train_per_step=0.1512.ckpt --train-loss-type smooth_l1_loss > ET-PCQM4MV2_dih_var0.04_var2_com_re_long_dw0.2_qm9_zpve_train_loss_type_smooth_l1_loss_finetuning.log 


python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var2_com_re_long_qm9_zpve_dw0.2_finetuning --dataset-arg zpve --pretrained-model /share/project/sharefs-skfeng/pre-training-via-denoising/experiments/pretrain_models/ET-PCQM4MV2_dih_var0.04_var2_com_re/step=399999-epoch=8-val_loss=0.1422-test_loss=0.2784-train_per_step=0.1512.ckpt  > ET-PCQM4MV2_dih_var0.04_var2_com_re_long_qm9_zpve_dw0.2_finetuning.log

python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.1_long.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var2_com_re_long_qm9_zpve_dw0.1_finetuning --dataset-arg zpve --pretrained-model /share/project/sharefs-skfeng/pre-training-via-denoising/experiments/pretrain_models/ET-PCQM4MV2_dih_var0.04_var2_com_re/step=399999-epoch=8-val_loss=0.1422-test_loss=0.2784-train_per_step=0.1512.ckpt  > ET-PCQM4MV2_dih_var0.04_var2_com_re_long_qm9_zpve_dw0.1_finetuning.log



python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var2_com_re_qm9_zpve_stand_false_finetuning --dataset-arg zpve --pretrained-model /share/project/sharefs-skfeng/pre-training-via-denoising/experiments/pretrain_models/ET-PCQM4MV2_dih_var0.04_var2_com_re/step=399999-epoch=8-val_loss=0.1422-test_loss=0.2784-train_per_step=0.1512.ckpt --standardize false > ET-PCQM4MV2_dih_var0.04_var2_com_re_qm9_zpve_stand_false_finetuning.log


python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id ET-PCQM4MV2_dih_var0.04_var2_com_re_qm9_gap_stand_false_finetuning --dataset-arg gap --pretrained-model /share/project/sharefs-skfeng/pre-training-via-denoising/experiments/pretrain_models/ET-PCQM4MV2_dih_var0.04_var2_com_re/step=399999-epoch=8-val_loss=0.1422-test_loss=0.2784-train_per_step=0.1512.ckpt --standardize false > ET-PCQM4MV2_dih_var0.04_var2_com_re_qm9_gap_stand_false_finetuning.log
