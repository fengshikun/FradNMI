# unify pre-training

CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id frad_baseline_diffusion_model_unify_pretrain_homo --dataset-arg homo --pretrained-model /mnt/nfs-ssd/data/fengshikun/e3_diffusion_for_molecules/outputs/pcq_torchmd_pretrain_4gpu_resume_resume/generative_model_92.npy --denoising-weight 0.1  --bond-length-scale 0.0  > frad_baseline_diffusion_model_unify_pretrain_homo.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id frad_baseline_diffusion_model_unify_pretrain_lumo --dataset-arg lumo --pretrained-model /mnt/nfs-ssd/data/fengshikun/e3_diffusion_for_molecules/outputs/pcq_torchmd_pretrain_4gpu_resume_resume/generative_model_92.npy --denoising-weight 0.1  --bond-length-scale 0.0  > frad_baseline_diffusion_model_unify_pretrain_lumo.log 2>&1 &



CUDA_VISIBLE_DEVICES=2 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id frad_baseline_diffusion_model_unify_pretrain_gap --dataset-arg gap --pretrained-model /mnt/nfs-ssd/data/fengshikun/e3_diffusion_for_molecules/outputs/pcq_torchmd_pretrain_4gpu_resume_resume/generative_model_92.npy --train-loss-type smooth_l1_loss  --bond-length-scale 0.0  > frad_baseline_diffusion_model_unify_pretrain_gap.log 2>&1 &


##### only denoise

CUDA_VISIBLE_DEVICES=3 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id frad_baseline_diffusion_model_only_diff_homo --dataset-arg homo --pretrained-model /mnt/nfs-ssd/data/fengshikun/e3_diffusion_for_molecules/outputs/pcq_torchmd_pretrain_4gpu_only_denoising/generative_model_18.npy --denoising-weight 0.1  --bond-length-scale 0.0  > frad_baseline_diffusion_model_only_diff_homo.log 2>&1 &


CUDA_VISIBLE_DEVICES=4 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id frad_baseline_diffusion_model_only_diff_lumo --dataset-arg lumo --pretrained-model /mnt/nfs-ssd/data/fengshikun/e3_diffusion_for_molecules/outputs/pcq_torchmd_pretrain_4gpu_only_denoising/generative_model_18.npy --denoising-weight 0.1  --bond-length-scale 0.0  > frad_baseline_diffusion_model_only_diff_lumo.log 2>&1 &



CUDA_VISIBLE_DEVICES=5 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id frad_baseline_diffusion_model_only_diff_gap --dataset-arg gap --pretrained-model /mnt/nfs-ssd/data/fengshikun/e3_diffusion_for_molecules/outputs/pcq_torchmd_pretrain_4gpu_only_denoising/generative_model_18.npy --train-loss-type smooth_l1_loss  --bond-length-scale 0.0  > frad_baseline_diffusion_model_only_diff_gap.log 2>&1 &


# without load the pos normalizer:

CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id frad_baseline_diffusion_model_unify_pretrain_homo_won --dataset-arg homo --pretrained-model /mnt/nfs-ssd/data/fengshikun/e3_diffusion_for_molecules/outputs/pcq_torchmd_pretrain_4gpu_resume_resume/generative_model_92.npy --denoising-weight 0.1  --bond-length-scale 0.0  > frad_baseline_diffusion_model_unify_pretrain_homo_won.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id frad_baseline_diffusion_model_unify_pretrain_lumo_won --dataset-arg lumo --pretrained-model /mnt/nfs-ssd/data/fengshikun/e3_diffusion_for_molecules/outputs/pcq_torchmd_pretrain_4gpu_resume_resume/generative_model_92.npy --denoising-weight 0.1  --bond-length-scale 0.0  > frad_baseline_diffusion_model_unify_pretrain_lumo_won.log 2>&1 &



CUDA_VISIBLE_DEVICES=2 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id frad_baseline_diffusion_model_unify_pretrain_gap_won --dataset-arg gap --pretrained-model /mnt/nfs-ssd/data/fengshikun/e3_diffusion_for_molecules/outputs/pcq_torchmd_pretrain_4gpu_resume_resume/generative_model_92.npy --train-loss-type smooth_l1_loss  --bond-length-scale 0.0  > frad_baseline_diffusion_model_unify_pretrain_gap_won.log 2>&1 &


# without loading noise head and pos normalizer

CUDA_VISIBLE_DEVICES=3 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id frad_baseline_diffusion_model_unify_pretrain_homo_wnn --dataset-arg homo --pretrained-model /mnt/nfs-ssd/data/fengshikun/e3_diffusion_for_molecules/outputs/pcq_torchmd_pretrain_4gpu_resume_resume/generative_model_92.npy --denoising-weight 0.1  --bond-length-scale 0.0  > frad_baseline_diffusion_model_unify_pretrain_homo_wnn.log 2>&1 &


CUDA_VISIBLE_DEVICES=4 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id frad_baseline_diffusion_model_unify_pretrain_lumo_wnn --dataset-arg lumo --pretrained-model /mnt/nfs-ssd/data/fengshikun/e3_diffusion_for_molecules/outputs/pcq_torchmd_pretrain_4gpu_resume_resume/generative_model_92.npy --denoising-weight 0.1  --bond-length-scale 0.0  > frad_baseline_diffusion_model_unify_pretrain_lumo_wnn.log 2>&1 &



CUDA_VISIBLE_DEVICES=5 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id frad_baseline_diffusion_model_unify_pretrain_gap_wnn --dataset-arg gap --pretrained-model /mnt/nfs-ssd/data/fengshikun/e3_diffusion_for_molecules/outputs/pcq_torchmd_pretrain_4gpu_resume_resume/generative_model_92.npy --train-loss-type smooth_l1_loss  --bond-length-scale 0.0  > frad_baseline_diffusion_model_unify_pretrain_gap_wnn.log 2>&1 &