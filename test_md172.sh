CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py --conf examples/ET-MD17_FT-angle_9500.yaml  --job-id ET-PCQM4MV2_var0.04_var2_com_re_md17_md17_aspirin_dihedral_angle_noise_scale_20_position_noise_scale_0.005_composition_true_sep_noisy_node_true_train_loss_type_smooth_l1_loss_finetuning --dataset-arg aspirin --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_var0.04_var2_com_re_md17/step=399999-epoch=8-val_loss=0.1712-test_loss=0.1653-train_per_step=0.1605.ckpt --dihedral-angle-noise-scale 20 --position-noise-scale 0.005 --composition true --sep-noisy-node true --train-loss-type smooth_l1_loss > ET-PCQM4MV2_var0.04_var2_com_re_md17_qm9_aspirin_dihedral_angle_noise_scale_20_position_noise_scale_0.005_composition_true_sep_noisy_node_true_train_loss_type_smooth_l1_loss_finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 python -u scripts/train.py --conf examples/ET-MD17_FT-angle_9500.yaml  --job-id ET-PCQM4MV2_var0.04_var2_com_re_md17_md17_ethanol_dihedral_angle_noise_scale_20_position_noise_scale_0.005_composition_true_sep_noisy_node_true_train_loss_type_smooth_l1_loss_finetuning --dataset-arg ethanol --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_var0.04_var2_com_re_md17/step=399999-epoch=8-val_loss=0.1712-test_loss=0.1653-train_per_step=0.1605.ckpt --dihedral-angle-noise-scale 20 --position-noise-scale 0.005 --composition true --sep-noisy-node true --train-loss-type smooth_l1_loss > ET-PCQM4MV2_var0.04_var2_com_re_md17_qm9_ethanol_dihedral_angle_noise_scale_20_position_noise_scale_0.005_composition_true_sep_noisy_node_true_train_loss_type_smooth_l1_loss_finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=2 python -u scripts/train.py --conf examples/ET-MD17_FT-angle_9500.yaml  --job-id ET-PCQM4MV2_var0.04_var1_com_re_md17_md17_aspirin_dihedral_angle_noise_scale_20_position_noise_scale_0.005_composition_true_sep_noisy_node_true_train_loss_type_smooth_l1_loss_finetuning --dataset-arg aspirin --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_var0.04_var1_com_re_md17/step=399999-epoch=8-val_loss=0.1615-test_loss=0.1591-train_per_step=0.1574.ckpt --dihedral-angle-noise-scale 20 --position-noise-scale 0.005 --composition true --sep-noisy-node true --train-loss-type smooth_l1_loss > ET-PCQM4MV2_var0.04_var1_com_re_md17_qm9_aspirin_dihedral_angle_noise_scale_20_position_noise_scale_0.005_composition_true_sep_noisy_node_true_train_loss_type_smooth_l1_loss_finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=3 python -u scripts/train.py --conf examples/ET-MD17_FT-angle_9500.yaml  --job-id ET-PCQM4MV2_var0.04_var1_com_re_md17_md17_ethanol_dihedral_angle_noise_scale_20_position_noise_scale_0.005_composition_true_sep_noisy_node_true_train_loss_type_smooth_l1_loss_finetuning --dataset-arg ethanol --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_var0.04_var1_com_re_md17/step=399999-epoch=8-val_loss=0.1615-test_loss=0.1591-train_per_step=0.1574.ckpt --dihedral-angle-noise-scale 20 --position-noise-scale 0.005 --composition true --sep-noisy-node true --train-loss-type smooth_l1_loss > ET-PCQM4MV2_var0.04_var1_com_re_md17_qm9_ethanol_dihedral_angle_noise_scale_20_position_noise_scale_0.005_composition_true_sep_noisy_node_true_train_loss_type_smooth_l1_loss_finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=4 python -u scripts/train.py --conf examples/ET-MD17_FT-angle_9500.yaml  --job-id ET-PCQM4MV2_var0.4_var2_com_re_md17_md17_aspirin_dihedral_angle_noise_scale_20_position_noise_scale_0.005_composition_true_sep_noisy_node_true_train_loss_type_smooth_l1_loss_finetuning --dataset-arg aspirin --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_var0.4_var2_com_re_md17/step=399999-epoch=8-val_loss=0.2371-test_loss=0.1970-train_per_step=0.2063.ckpt --dihedral-angle-noise-scale 20 --position-noise-scale 0.005 --composition true --sep-noisy-node true --train-loss-type smooth_l1_loss > ET-PCQM4MV2_var0.4_var2_com_re_md17_qm9_aspirin_dihedral_angle_noise_scale_20_position_noise_scale_0.005_composition_true_sep_noisy_node_true_train_loss_type_smooth_l1_loss_finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=5 python -u scripts/train.py --conf examples/ET-MD17_FT-angle_9500.yaml  --job-id ET-PCQM4MV2_var0.4_var2_com_re_md17_md17_ethanol_dihedral_angle_noise_scale_20_position_noise_scale_0.005_composition_true_sep_noisy_node_true_train_loss_type_smooth_l1_loss_finetuning --dataset-arg ethanol --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_var0.4_var2_com_re_md17/step=399999-epoch=8-val_loss=0.2371-test_loss=0.1970-train_per_step=0.2063.ckpt --dihedral-angle-noise-scale 20 --position-noise-scale 0.005 --composition true --sep-noisy-node true --train-loss-type smooth_l1_loss > ET-PCQM4MV2_var0.4_var2_com_re_md17_qm9_ethanol_dihedral_angle_noise_scale_20_position_noise_scale_0.005_composition_true_sep_noisy_node_true_train_loss_type_smooth_l1_loss_finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=6 python -u scripts/train.py --conf examples/ET-MD17_FT-angle_9500.yaml  --job-id ET-PCQM4MV2_var0.004_var2_com_re_md17_md17_aspirin_dihedral_angle_noise_scale_20_position_noise_scale_0.005_composition_true_sep_noisy_node_true_train_loss_type_smooth_l1_loss_finetuning --dataset-arg aspirin --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_var0.004_var2_com_re_md17/step=399999-epoch=8-val_loss=0.3123-test_loss=0.2975-train_per_step=0.3015.ckpt --dihedral-angle-noise-scale 20 --position-noise-scale 0.005 --composition true --sep-noisy-node true --train-loss-type smooth_l1_loss > ET-PCQM4MV2_var0.004_var2_com_re_md17_qm9_aspirin_dihedral_angle_noise_scale_20_position_noise_scale_0.005_composition_true_sep_noisy_node_true_train_loss_type_smooth_l1_loss_finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=7 python -u scripts/train.py --conf examples/ET-MD17_FT-angle_9500.yaml  --job-id ET-PCQM4MV2_var0.004_var2_com_re_md17_md17_ethanol_dihedral_angle_noise_scale_20_position_noise_scale_0.005_composition_true_sep_noisy_node_true_train_loss_type_smooth_l1_loss_finetuning --dataset-arg ethanol --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_var0.004_var2_com_re_md17/step=399999-epoch=8-val_loss=0.3123-test_loss=0.2975-train_per_step=0.3015.ckpt --dihedral-angle-noise-scale 20 --position-noise-scale 0.005 --composition true --sep-noisy-node true --train-loss-type smooth_l1_loss > ET-PCQM4MV2_var0.004_var2_com_re_md17_qm9_ethanol_dihedral_angle_noise_scale_20_position_noise_scale_0.005_composition_true_sep_noisy_node_true_train_loss_type_smooth_l1_loss_finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=8 python -u scripts/train.py --conf examples/ET-MD17_FT-angle_9500.yaml  --job-id ET-PCQM4MV2_var0.04_var20_com_re_md17_md17_aspirin_dihedral_angle_noise_scale_20_position_noise_scale_0.005_composition_true_sep_noisy_node_true_train_loss_type_smooth_l1_loss_finetuning --dataset-arg aspirin --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_var0.04_var20_com_re_md17/step=399999-epoch=8-val_loss=0.2095-test_loss=0.1942-train_per_step=0.1871.ckpt --dihedral-angle-noise-scale 20 --position-noise-scale 0.005 --composition true --sep-noisy-node true --train-loss-type smooth_l1_loss > ET-PCQM4MV2_var0.04_var20_com_re_md17_qm9_aspirin_dihedral_angle_noise_scale_20_position_noise_scale_0.005_composition_true_sep_noisy_node_true_train_loss_type_smooth_l1_loss_finetuning.log 2>&1 &


CUDA_VISIBLE_DEVICES=9 python -u scripts/train.py --conf examples/ET-MD17_FT-angle_9500.yaml  --job-id ET-PCQM4MV2_var0.04_var20_com_re_md17_md17_ethanol_dihedral_angle_noise_scale_20_position_noise_scale_0.005_composition_true_sep_noisy_node_true_train_loss_type_smooth_l1_loss_finetuning --dataset-arg ethanol --pretrained-model /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_var0.04_var20_com_re_md17/step=399999-epoch=8-val_loss=0.2095-test_loss=0.1942-train_per_step=0.1871.ckpt --dihedral-angle-noise-scale 20 --position-noise-scale 0.005 --composition true --sep-noisy-node true --train-loss-type smooth_l1_loss > ET-PCQM4MV2_var0.04_var20_com_re_md17_qm9_ethanol_dihedral_angle_noise_scale_20_position_noise_scale_0.005_composition_true_sep_noisy_node_true_train_loss_type_smooth_l1_loss_finetuning.log 2>&1 &


