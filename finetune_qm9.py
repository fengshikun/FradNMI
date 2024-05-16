# python finetune_on_qm9.py --pretrain_model /home/AI4Science/fengsk/Pretraining-Denoising/experiments/ET-PCQM4MV2_var0.04/step=399999-epoch=8-val_loss=0.1241-test_loss=0.2535-train_per_step=0.1218.ckpt --job_prefix pretrain_baseline_var0.04;

#python finetune_on_qm9.py --pretrain_model /home/AI4Science/fengsk/Model_bak/pretrain_models/f2d_sample128_frad_multask_Areg/step=183143-epoch=7-val_loss=0.2628-test_loss=0.2585-train_per_step=0.2517.ckpt --job_prefix f2d_sample128_frad_multask_Areg 

# python finetune_on_qm9.py --pretrain_model /home/AI4Science/fengsk/Model_bak/pretrain_models/DeCL/pretraining_f2d_pos_frad_debug_ew0.1_pna_mask0.15/step=249999-epoch=8-val_loss=6678168.0000-test_loss=384682.5938-train_per_step=7.7368.ckpt --job_prefix pretraining_f2d_pos_frad_debug_ew0.1_pna_mask0.15

# python finetune_on_qm9.py --pretrain_model /home/AI4Science/fengsk/Model_bak/pretrain_models/DeCL/pretraining_f2d_pos_frad_ctl_gf_frag_mask0.2_unipcq/step=735829-epoch=4-val_loss=0.2124-test_loss=0.2811-train_per_step=0.2378.ckpt --job_prefix deCL_epoch4


# python finetune_qm9.py --pretrain_model /data/protein/SKData/Frad_NMI/FradNMI/experiments/frad_pretraining_10w/step=11407-epoch=7-val_loss=0.2065-test_loss=0.2143-train_per_step=0.2014.ckpt --job_prefix frad_pretraining_10w --start_gid 3


# python finetune_qm9.py --pretrain_model /data/protein/SKData/Frad_NMI/FradNMI/experiments/frad_pretraining_rdkit_10w/step=11407-epoch=7-val_loss=0.2023-test_loss=0.2045-train_per_step=0.1883.ckpt --job_prefix frad_pretraining_rdkit_10w --start_gid 0
import os
import argparse







qm9_task = {'dipole_moment': 0, 'isotropic_polarizability': 1, 'homo': 2, 'lumo': 3, 'gap': 4, 'electronic_spatial_extent': 5, 'zpve': 6, 'energy_U0': 7, 'energy_U': 8, 'enthalpy_H': 9, 'free_energy': 10, 'heat_capacity': 11}







# cmd_prefix = 'srun --mem=64000 --gres=gpu:1 --time 6-12:00:00 --job-name test_qm9'

# cmd_suffix = '--model equivariant-transformerf2d --bond-length-scale 0.0 --dataset-root /home/AI4Science/fengsk/Denoise_Data/qm9'


# qm9_task = {'isotropic_polarizability': 1, 'gap': 4, 'electronic_spatial_extent': 5, 'zpve': 6, 'energy_U0': 7, 'energy_U': 8, 'enthalpy_H': 9, 'free_energy': 10, 'heat_capacity': 11}

# qm9_task = {'enthalpy_H': 9}

# qm9_task = {'isotropic_polarizability': 1}

cmd_prefix = 'CUDA_VISIBLE_DEVICES={}'

# cmd_suffix = '--batch-size 64 --lr-warmup-steps 20000 --lr-cosine-length 1000000 --num-steps 1000000  --model equivariant-transformerf2d --bond-length-scale 0.0 --dataset-root /home/AI4Science/fengsk/Denoise_Data/qm9'

cmd_suffix = ''


QM9_CMD = [
    '{} python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id {}_lumo --dataset-arg lumo --pretrained-model {} --denoising-weight 0.1 {} > {}.log 2>&1 &',
    
    '{} python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id {}_homo --dataset-arg homo --pretrained-model {} --denoising-weight 0.1 {} > {}.log 2>&1 &',
    
    '{} python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id {}_gap --dataset-arg gap --pretrained-model {} --train-loss-type smooth_l1_loss {} > {}.log 2>&1 &',
    
]

# qm9_task = {'homo': 2, 'gap': 4}
# qm9_task = {'lumo': 3}

qm9_task = {'lumo': 0, 'homo': 1, 'gap': 2}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument("--pretrain_model", type=str, default="sce")
    parser.add_argument("--job_prefix", type=str, default="job_prefix")
    
    parser.add_argument("--job_suffix", type=str, default="")
    
    parser.add_argument("--start_gid", type=int, default=0)
    
    args = parser.parse_args()
    pretrain_model = args.pretrain_model
    job_prefix = args.job_prefix
    
    start_gid = args.start_gid

    cmd_suffix = args.job_suffix
    
    for task in qm9_task:
        # base_cmd = f'srun --mem=64000 --gres=gpu:1 --time 6-12:00:00 --job-name {job_prefix}_qm9_{task}_finetuning python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id {job_prefix}_qm9_{task}_finetuning --dataset-arg {task} --pretrained-model {pretrain_model} > {job_prefix}_qm9_{task}_finetuning.log 2>&1 &'

        # base_cmd = f'srun --mem=64000 --gres=gpu:a100-80G --time 6-12:00:00 --job-name {job_prefix}_qm9_{task}_finetuning python -u scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id {job_prefix}_qm9_{task}_finetuning --dataset-arg {task} --pretrained-model {pretrain_model} > {job_prefix}_qm9_{task}_finetuning.log 2>&1 &'
        
        base_cmd = QM9_CMD[qm9_task[task]]
        
        cmd_prefix1 = cmd_prefix.format(start_gid)
        
        cmd_suffix1 = cmd_suffix + ' --dataset-root /data/protein/SKData/DenoisingData/qm9'
        
        exe_cmd = base_cmd.format(cmd_prefix1, job_prefix, pretrain_model, cmd_suffix1, f'{job_prefix}_{task}')
        start_gid += 1
        # import pdb; pdb.set_trace()
        print(exe_cmd + '\n')
        # os.system(exe_cmd)
    pass