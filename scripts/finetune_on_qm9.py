# python finetune_on_qm9.py --pretrain_model /home/AI4Science/fengsk/Pretraining-Denoising/experiments/ET-PCQM4MV2_var0.04/step=399999-epoch=8-val_loss=0.1241-test_loss=0.2535-train_per_step=0.1218.ckpt --job_prefix pretrain_baseline_var0.04;

#python finetune_on_qm9.py --pretrain_model /home/AI4Science/fengsk/Model_bak/pretrain_models/f2d_sample128_frad_multask_Areg/step=183143-epoch=7-val_loss=0.2628-test_loss=0.2585-train_per_step=0.2517.ckpt --job_prefix f2d_sample128_frad_multask_Areg 

import os
import argparse







qm9_task = {'dipole_moment': 0, 'isotropic_polarizability': 1, 'lumo': 2, 'homo': 3, 'gap': 4, 'electronic_spatial_extent': 5, 'zpve': 6, 'energy_U0': 7, 'energy_U': 8, 'enthalpy_H': 9, 'free_energy': 10, 'heat_capacity': 11}







# cmd_prefix = 'srun --mem=64000 --gres=gpu:1 --time 6-12:00:00 --job-name test_qm9'

# cmd_suffix = '--model equivariant-transformerf2d --bond-length-scale 0.0 --dataset-root /home/AI4Science/fengsk/Denoise_Data/qm9'


# qm9_task = {'isotropic_polarizability': 1, 'gap': 4, 'electronic_spatial_extent': 5, 'zpve': 6, 'energy_U0': 7, 'energy_U': 8, 'enthalpy_H': 9, 'free_energy': 10, 'heat_capacity': 11}

# qm9_task = {'enthalpy_H': 9}

# qm9_task = {'isotropic_polarizability': 1}

# python finetune_on_qm9.py --pretrain_model /home/AI4Science/niyy/DeCL/step=735829-epoch=4-val_loss=0.2124-test_loss=0.2811-train_per_step=0.2378.ckpt --job_prefix deCL_epoch4
# python finetune_on_qm9.py --pretrain_model /home/AI4Science/fengsk/Model_bak/pretrain_models/DeCL/pretraining_f2d_pos_frad_ctl_gf_frag_mask0.2_unipcq/step=1177327-epoch=7-val_loss=0.2735-test_loss=0.2023-train_per_step=0.2393.ckpt --job_prefix deCL_epoch7

# python finetune_on_qm9.py --pretrain_model /home/AI4Science/fengsk/Model_bak/pretrain_models/DeCL/pretraining_f2d_pos_frad_ctl_gf_frag_mask0.2_unipcq/step=735829-epoch=4-val_loss=0.2124-test_loss=0.2811-train_per_step=0.2378.ckpt --job_prefix deCL_epoch4

# python finetune_on_qm9.py --pretrain_model /home/admin01/FradNMI/experiments/frad_pretraining/step=386103-epoch=7-val_loss=0.1509-test_loss=0.1408-train_per_step=0.1309.ckpt --job_prefix frad_baseline

# python finetune_on_qm9.py --pretrain_model /home/admin01/FradNMI/experiments/frad_pretraining_emb512_num_12/step=375383-epoch=7-val_loss=0.1347-test_loss=0.1379-train_per_step=0.1471.ckpt --job_prefix frad_pretraining_emb512_num_12

# python finetune_on_qm9.py --pretrain_model /home/admin01/FradNMI/experiments/frad_pretraining_num_12/step=386103-epoch=7-val_loss=0.1473-test_loss=0.1354-train_per_step=0.1256.ckpt --job_prefix frad_pretraining_num_12



cmd_prefix = 'srun --mem=64000 --gres=gpu:1 --time 6-12:00:00 --job-name test_qm9'

cmd_suffix = '--batch-size 64 --lr-warmup-steps 20000 --lr-cosine-length 1000000 --num-steps 1000000  --model equivariant-transformerf2d --bond-length-scale 0.0 --dataset-root /home/AI4Science/niyy/ShikunData/Data/qm9'



QM9_CMD = [
    
    '{} python -u scripts/train.py --conf examples/ET-QM9-FT_D.yaml --layernorm-on-vec whitened --job-id {}_dipole_moment --dataset-arg dipole_moment --pretrained-model {} {} > {}.log 2>&1 &',
    
    '{} python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id {}_isotropic_polarizability --dataset-arg isotropic_polarizability --pretrained-model {} --train-loss-type smooth_l1_loss {} > {}.log 2>&1 &',
    
    '{} python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long_f2d.yaml --layernorm-on-vec whitened --job-id {}_lumo --dataset-arg lumo --pretrained-model {} --denoising-weight 0.1 {} > {}.log 2>&1 &',
    
    '{} python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long_f2d.yaml --layernorm-on-vec whitened --job-id {}_homo --dataset-arg homo --pretrained-model {} --denoising-weight 0.1 {} > {}.log 2>&1 &',
    
    '{} python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long_f2d.yaml --layernorm-on-vec whitened --job-id {}_gap --dataset-arg gap --pretrained-model {} --train-loss-type smooth_l1_loss {} > {}.log 2>&1 &',
    
    '{} python -u scripts/train.py --conf examples/ET-QM9-FT_E.yaml --layernorm-on-vec whitened --job-id {}_electronic_spatial_extent --dataset-arg electronic_spatial_extent --pretrained-model {} --output-model ElectronicSpatialExtent2 {} > {}.log 2>&1 &',
    
    '{} python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.1_long.yaml --layernorm-on-vec whitened --job-id {}_zpve --dataset-arg zpve --pretrained-model {} {} > {}.log 2>&1 &',
    
    '{} python -u scripts/train.py --conf examples/ET-QM9-FT-nt_dw_0.2_long_f2d.yaml --layernorm-on-vec whitened --job-id {}_energy_U0 --dataset-arg energy_U0 --pretrained-model {} {} > {}.log 2>&1 &',
    
    '{} python -u scripts/train.py --conf examples/ET-QM9-FT-nt_dw_0.2_long_f2d.yaml --layernorm-on-vec whitened --job-id {}_energy_U --dataset-arg energy_U --pretrained-model {} {} > {}.log 2>&1 &',
    
    '{} python -u scripts/train.py --conf examples/ET-QM9-FT-nt_dw_0.2_long_f2d.yaml --layernorm-on-vec whitened --job-id {}_enthalpy_H --dataset-arg enthalpy_H --pretrained-model {} {} > {}.log 2>&1 &',
    
    '{} python -u scripts/train.py --conf examples/ET-QM9-FT-nt_dw_0.2_long_f2d.yaml --layernorm-on-vec whitened --job-id {}_free_energy --dataset-arg free_energy --pretrained-model {} {} > {}.log 2>&1 &',
    
    
    '{} python -u scripts/train.py --conf examples/ET-QM9-FT-nt.yaml --layernorm-on-vec whitened --job-id {}_heat_capacity --dataset-arg heat_capacity --pretrained-model {} {} > {}.log 2>&1 &',
    
]

qm9_task = {'lumo': 2, 'homo': 3, 'gap': 4}
# qm9_task = {'homo': 2, 'gap': 4}
# qm9_task = {'lumo': 3}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument("--pretrain_model", type=str, default="sce")
    parser.add_argument("--job_prefix", type=str, default="job_prefix")
    args = parser.parse_args()
    pretrain_model = args.pretrain_model
    job_prefix = args.job_prefix

    for task in qm9_task:
        # base_cmd = f'srun --mem=64000 --gres=gpu:1 --time 6-12:00:00 --job-name {job_prefix}_qm9_{task}_finetuning python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id {job_prefix}_qm9_{task}_finetuning --dataset-arg {task} --pretrained-model {pretrain_model} > {job_prefix}_qm9_{task}_finetuning.log 2>&1 &'

        # base_cmd = f'srun --mem=64000 --gres=gpu:a100-80G --time 6-12:00:00 --job-name {job_prefix}_qm9_{task}_finetuning python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id {job_prefix}_qm9_{task}_finetuning --dataset-arg {task} --pretrained-model {pretrain_model} > {job_prefix}_qm9_{task}_finetuning.log 2>&1 &'
        
        base_cmd = QM9_CMD[qm9_task[task]]
        
        exe_cmd = base_cmd.format(cmd_prefix, job_prefix, pretrain_model, cmd_suffix, f'{job_prefix}_{task}')
        
        
        print(exe_cmd + '\n')
        # os.system(exe_cmd)
    pass