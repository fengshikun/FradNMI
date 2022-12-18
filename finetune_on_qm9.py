# python finetune_on_qm9.py --pretrain_model /home/AI4Science/fengsk/Pretraining-Denoising/experiments/ET-PCQM4MV2_var0.04/step=399999-epoch=8-val_loss=0.1241-test_loss=0.2535-train_per_step=0.1218.ckpt --job_prefix pretrain_baseline_var0.04;

import os
import argparse





qm9_task = {'dipole_moment': 0, 'isotropic_polarizability': 1, 'homo': 2, 'lumo': 3, 'gap': 4, 'electronic_spatial_extent': 5, 'zpve': 6, 'energy_U0': 7, 'energy_U': 8, 'enthalpy_H': 9, 'free_energy': 10, 'heat_capacity': 11}


# qm9_task = {'homo': 2, 'gap': 4}
qm9_task = {'lumo': 3}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument("--pretrain_model", type=str, default="sce")
    parser.add_argument("--job_prefix", type=str, default="job_prefix")
    args = parser.parse_args()
    pretrain_model = args.pretrain_model
    job_prefix = args.job_prefix

    for task in qm9_task:
        base_cmd = f'srun --mem=64000 --gres=gpu:1 --time 6-12:00:00 --job-name {job_prefix}_qm9_{task}_finetuning python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id {job_prefix}_qm9_{task}_finetuning --dataset-arg {task} --pretrained-model {pretrain_model} > {job_prefix}_qm9_{task}_finetuning.log 2>&1 &'

        # base_cmd = f'srun --mem=64000 --gres=gpu:a100-80G --time 6-12:00:00 --job-name {job_prefix}_qm9_{task}_finetuning python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id {job_prefix}_qm9_{task}_finetuning --dataset-arg {task} --pretrained-model {pretrain_model} > {job_prefix}_qm9_{task}_finetuning.log 2>&1 &'
        os.system(base_cmd)
    pass