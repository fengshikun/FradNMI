CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py --conf examples/ET-MD17_FT-angle_9500.yaml  --job-id ET-PCQM4MV2_var0.04_var2_com_re_md17_md17_naphthalene_dihedral_angle_noise_scale_20_position_noise_scale_0.005_composition_true_sep_noisy_node_true_train_loss_type_smooth_l1_loss_finetuning_seed1 --dataset-arg naphthalene --pretrained-model /share/project/sharefs-skfeng/pre-training-via-denoising/experiments/pretrain_models/ET-PCQM4MV2_var0.04_var2_com_re_md17/step=399999-epoch=8-val_loss=0.1626-test_loss=0.2945-train_per_step=0.1681.ckpt --dihedral-angle-noise-scale 20 --position-noise-scale 0.005 --composition true --sep-noisy-node true --train-loss-type smooth_l1_loss --seed 1 > ET-PCQM4MV2_var0.04_var2_com_re_md17_qm9_naphthalene_dihedral_angle_noise_scale_20_position_noise_scale_0.005_composition_true_sep_noisy_node_true_train_loss_type_smooth_l1_loss_finetuning_seed1.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 python -u scripts/train.py --conf examples/ET-MD17_FT-angle_9500.yaml  --job-id ET-PCQM4MV2_var0.04_var2_com_re_md17_md17_ethanol_dihedral_angle_noise_scale_20_position_noise_scale_0.005_composition_true_sep_noisy_node_true_train_loss_type_smooth_l1_loss_finetuning_seed1 --dataset-arg ethanol --pretrained-model /share/project/sharefs-skfeng/pre-training-via-denoising/experiments/pretrain_models/ET-PCQM4MV2_var0.04_var2_com_re_md17/step=399999-epoch=8-val_loss=0.1626-test_loss=0.2945-train_per_step=0.1681.ckpt --dihedral-angle-noise-scale 20 --position-noise-scale 0.005 --composition true --sep-noisy-node true --train-loss-type smooth_l1_loss --seed 1 > ET-PCQM4MV2_var0.04_var2_com_re_md17_qm9_ethanol_dihedral_angle_noise_scale_20_position_noise_scale_0.005_composition_true_sep_noisy_node_true_train_loss_type_smooth_l1_loss_finetuning_seed1.log 2>&1 &


CUDA_VISIBLE_DEVICES=2 python -u scripts/train.py --conf examples/ET-MD17_FT-angle_9500.yaml  --job-id ET-PCQM4MV2_var0.04_var2_com_re_md17_md17_aspirin_dihedral_angle_noise_scale_20_position_noise_scale_0.005_composition_true_sep_noisy_node_true_train_loss_type_smooth_l1_loss_finetuning_seed1 --dataset-arg aspirin --pretrained-model /share/project/sharefs-skfeng/pre-training-via-denoising/experiments/pretrain_models/ET-PCQM4MV2_var0.04_var2_com_re_md17/step=399999-epoch=8-val_loss=0.1626-test_loss=0.2945-train_per_step=0.1681.ckpt --dihedral-angle-noise-scale 20 --position-noise-scale 0.005 --composition true --sep-noisy-node true --train-loss-type smooth_l1_loss --seed 1 > ET-PCQM4MV2_var0.04_var2_com_re_md17_qm9_aspirin_dihedral_angle_noise_scale_20_position_noise_scale_0.005_composition_true_sep_noisy_node_true_train_loss_type_smooth_l1_loss_finetuning_seed1.log 2>&1 &


CUDA_VISIBLE_DEVICES=3 python -u scripts/train.py --conf examples/ET-MD17_FT-angle_9500.yaml  --job-id ET-PCQM4MV2_var0.04_var2_com_re_md17_md17_malonaldehyde_dihedral_angle_noise_scale_20_position_noise_scale_0.005_composition_true_sep_noisy_node_true_train_loss_type_smooth_l1_loss_finetuning_seed1 --dataset-arg malonaldehyde --pretrained-model /share/project/sharefs-skfeng/pre-training-via-denoising/experiments/pretrain_models/ET-PCQM4MV2_var0.04_var2_com_re_md17/step=399999-epoch=8-val_loss=0.1626-test_loss=0.2945-train_per_step=0.1681.ckpt --dihedral-angle-noise-scale 20 --position-noise-scale 0.005 --composition true --sep-noisy-node true --train-loss-type smooth_l1_loss --seed 1 > ET-PCQM4MV2_var0.04_var2_com_re_md17_qm9_malonaldehyde_dihedral_angle_noise_scale_20_position_noise_scale_0.005_composition_true_sep_noisy_node_true_train_loss_type_smooth_l1_loss_finetuning_seed1.log 2>&1 &


CUDA_VISIBLE_DEVICES=4 python -u scripts/train.py --conf examples/ET-MD17_FT-angle_9500.yaml  --job-id ET-PCQM4MV2_var0.04_var2_com_re_md17_md17_naphthalene_dihedral_angle_noise_scale_20_position_noise_scale_0.005_composition_true_sep_noisy_node_true_train_loss_type_smooth_l1_loss_finetuning_seed2 --dataset-arg naphthalene --pretrained-model /share/project/sharefs-skfeng/pre-training-via-denoising/experiments/pretrain_models/ET-PCQM4MV2_var0.04_var2_com_re_md17/step=399999-epoch=8-val_loss=0.1626-test_loss=0.2945-train_per_step=0.1681.ckpt --dihedral-angle-noise-scale 20 --position-noise-scale 0.005 --composition true --sep-noisy-node true --train-loss-type smooth_l1_loss --seed 2 > ET-PCQM4MV2_var0.04_var2_com_re_md17_qm9_naphthalene_dihedral_angle_noise_scale_20_position_noise_scale_0.005_composition_true_sep_noisy_node_true_train_loss_type_smooth_l1_loss_finetuning_seed2.log 2>&1 &


CUDA_VISIBLE_DEVICES=5 python -u scripts/train.py --conf examples/ET-MD17_FT-angle_9500.yaml  --job-id ET-PCQM4MV2_var0.04_var2_com_re_md17_md17_ethanol_dihedral_angle_noise_scale_20_position_noise_scale_0.005_composition_true_sep_noisy_node_true_train_loss_type_smooth_l1_loss_finetuning_seed2 --dataset-arg ethanol --pretrained-model /share/project/sharefs-skfeng/pre-training-via-denoising/experiments/pretrain_models/ET-PCQM4MV2_var0.04_var2_com_re_md17/step=399999-epoch=8-val_loss=0.1626-test_loss=0.2945-train_per_step=0.1681.ckpt --dihedral-angle-noise-scale 20 --position-noise-scale 0.005 --composition true --sep-noisy-node true --train-loss-type smooth_l1_loss --seed 2 > ET-PCQM4MV2_var0.04_var2_com_re_md17_qm9_ethanol_dihedral_angle_noise_scale_20_position_noise_scale_0.005_composition_true_sep_noisy_node_true_train_loss_type_smooth_l1_loss_finetuning_seed2.log 2>&1 &


CUDA_VISIBLE_DEVICES=6 python -u scripts/train.py --conf examples/ET-MD17_FT-angle_9500.yaml  --job-id ET-PCQM4MV2_var0.04_var2_com_re_md17_md17_aspirin_dihedral_angle_noise_scale_20_position_noise_scale_0.005_composition_true_sep_noisy_node_true_train_loss_type_smooth_l1_loss_finetuning_seed2 --dataset-arg aspirin --pretrained-model /share/project/sharefs-skfeng/pre-training-via-denoising/experiments/pretrain_models/ET-PCQM4MV2_var0.04_var2_com_re_md17/step=399999-epoch=8-val_loss=0.1626-test_loss=0.2945-train_per_step=0.1681.ckpt --dihedral-angle-noise-scale 20 --position-noise-scale 0.005 --composition true --sep-noisy-node true --train-loss-type smooth_l1_loss --seed 2 > ET-PCQM4MV2_var0.04_var2_com_re_md17_qm9_aspirin_dihedral_angle_noise_scale_20_position_noise_scale_0.005_composition_true_sep_noisy_node_true_train_loss_type_smooth_l1_loss_finetuning_seed2.log 2>&1 &


CUDA_VISIBLE_DEVICES=7 python -u scripts/train.py --conf examples/ET-MD17_FT-angle_9500.yaml  --job-id ET-PCQM4MV2_var0.04_var2_com_re_md17_md17_malonaldehyde_dihedral_angle_noise_scale_20_position_noise_scale_0.005_composition_true_sep_noisy_node_true_train_loss_type_smooth_l1_loss_finetuning_seed2 --dataset-arg malonaldehyde --pretrained-model /share/project/sharefs-skfeng/pre-training-via-denoising/experiments/pretrain_models/ET-PCQM4MV2_var0.04_var2_com_re_md17/step=399999-epoch=8-val_loss=0.1626-test_loss=0.2945-train_per_step=0.1681.ckpt --dihedral-angle-noise-scale 20 --position-noise-scale 0.005 --composition true --sep-noisy-node true --train-loss-type smooth_l1_loss --seed 2 > ET-PCQM4MV2_var0.04_var2_com_re_md17_qm9_malonaldehyde_dihedral_angle_noise_scale_20_position_noise_scale_0.005_composition_true_sep_noisy_node_true_train_loss_type_smooth_l1_loss_finetuning_seed2.log 2>&1 &


CUDA_VISIBLE_DEVICES=8 python -u scripts/train.py --conf examples/ET-MD17_FT-angle_9500.yaml  --job-id ET-PCQM4MV2_var0.04_var2_com_re_md17_md17_naphthalene_dihedral_angle_noise_scale_20_position_noise_scale_0.005_composition_true_sep_noisy_node_true_train_loss_type_smooth_l1_loss_finetuning_seed3 --dataset-arg naphthalene --pretrained-model /share/project/sharefs-skfeng/pre-training-via-denoising/experiments/pretrain_models/ET-PCQM4MV2_var0.04_var2_com_re_md17/step=399999-epoch=8-val_loss=0.1626-test_loss=0.2945-train_per_step=0.1681.ckpt --dihedral-angle-noise-scale 20 --position-noise-scale 0.005 --composition true --sep-noisy-node true --train-loss-type smooth_l1_loss --seed 3 > ET-PCQM4MV2_var0.04_var2_com_re_md17_qm9_naphthalene_dihedral_angle_noise_scale_20_position_noise_scale_0.005_composition_true_sep_noisy_node_true_train_loss_type_smooth_l1_loss_finetuning_seed3.log 2>&1 &


CUDA_VISIBLE_DEVICES=9 python -u scripts/train.py --conf examples/ET-MD17_FT-angle_9500.yaml  --job-id ET-PCQM4MV2_var0.04_var2_com_re_md17_md17_ethanol_dihedral_angle_noise_scale_20_position_noise_scale_0.005_composition_true_sep_noisy_node_true_train_loss_type_smooth_l1_loss_finetuning_seed3 --dataset-arg ethanol --pretrained-model /share/project/sharefs-skfeng/pre-training-via-denoising/experiments/pretrain_models/ET-PCQM4MV2_var0.04_var2_com_re_md17/step=399999-epoch=8-val_loss=0.1626-test_loss=0.2945-train_per_step=0.1681.ckpt --dihedral-angle-noise-scale 20 --position-noise-scale 0.005 --composition true --sep-noisy-node true --train-loss-type smooth_l1_loss --seed 3 > ET-PCQM4MV2_var0.04_var2_com_re_md17_qm9_ethanol_dihedral_angle_noise_scale_20_position_noise_scale_0.005_composition_true_sep_noisy_node_true_train_loss_type_smooth_l1_loss_finetuning_seed3.log 2>&1 &


CUDA_VISIBLE_DEVICES=10 python -u scripts/train.py --conf examples/ET-MD17_FT-angle_9500.yaml  --job-id ET-PCQM4MV2_var0.04_var2_com_re_md17_md17_aspirin_dihedral_angle_noise_scale_20_position_noise_scale_0.005_composition_true_sep_noisy_node_true_train_loss_type_smooth_l1_loss_finetuning_seed3 --dataset-arg aspirin --pretrained-model /share/project/sharefs-skfeng/pre-training-via-denoising/experiments/pretrain_models/ET-PCQM4MV2_var0.04_var2_com_re_md17/step=399999-epoch=8-val_loss=0.1626-test_loss=0.2945-train_per_step=0.1681.ckpt --dihedral-angle-noise-scale 20 --position-noise-scale 0.005 --composition true --sep-noisy-node true --train-loss-type smooth_l1_loss --seed 3 > ET-PCQM4MV2_var0.04_var2_com_re_md17_qm9_aspirin_dihedral_angle_noise_scale_20_position_noise_scale_0.005_composition_true_sep_noisy_node_true_train_loss_type_smooth_l1_loss_finetuning_seed3.log 2>&1 &


CUDA_VISIBLE_DEVICES=11 python -u scripts/train.py --conf examples/ET-MD17_FT-angle_9500.yaml  --job-id ET-PCQM4MV2_var0.04_var2_com_re_md17_md17_malonaldehyde_dihedral_angle_noise_scale_20_position_noise_scale_0.005_composition_true_sep_noisy_node_true_train_loss_type_smooth_l1_loss_finetuning_seed3 --dataset-arg malonaldehyde --pretrained-model /share/project/sharefs-skfeng/pre-training-via-denoising/experiments/pretrain_models/ET-PCQM4MV2_var0.04_var2_com_re_md17/step=399999-epoch=8-val_loss=0.1626-test_loss=0.2945-train_per_step=0.1681.ckpt --dihedral-angle-noise-scale 20 --position-noise-scale 0.005 --composition true --sep-noisy-node true --train-loss-type smooth_l1_loss --seed 3 > ET-PCQM4MV2_var0.04_var2_com_re_md17_qm9_malonaldehyde_dihedral_angle_noise_scale_20_position_noise_scale_0.005_composition_true_sep_noisy_node_true_train_loss_type_smooth_l1_loss_finetuning_seed3.log 2>&1 &


