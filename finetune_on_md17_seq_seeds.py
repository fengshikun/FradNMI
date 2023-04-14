import os
import argparse


# python finetune_on_md17_seq.py --pretrain_models /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_var0.04_var2_com_re_md17/step=399999-epoch=8-val_loss=0.1712-test_loss=0.1653-train_per_step=0.1605.ckpt /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_var0.04_var1_com_re_md17/step=399999-epoch=8-val_loss=0.1615-test_loss=0.1591-train_per_step=0.1574.ckpt /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_var0.4_var2_com_re_md17/step=399999-epoch=8-val_loss=0.2371-test_loss=0.1970-train_per_step=0.2063.ckpt /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_var0.004_var2_com_re_md17/step=399999-epoch=8-val_loss=0.3123-test_loss=0.2975-train_per_step=0.3015.ckpt /home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_var0.04_var20_com_re_md17/step=399999-epoch=8-val_loss=0.2095-test_loss=0.1942-train_per_step=0.1871.ckpt  --conf ET-MD17_FT-angle_9500.yaml --add_cmd "--dihedral-angle-noise-scale 20 --position-noise-scale 0.005 --composition true --sep-noisy-node true --train-loss-type smooth_l1_loss"

#  python finetune_on_md17_seq.py --pretrain_models /share/project/sharefs-skfeng/pre-training-via-denoising/experiments/pretrain_models/ET-PCQM4MV2_var0.04_var2_com_re_md17/step=399999-epoch=8-val_loss=0.1626-test_loss=0.2945-train_per_step=0.1681.ckpt --config ET-MD17_FT-angle.yaml --add_cmd="--dihedral-angle-noise-scale 20 --position-noise-scale 0.005 --composition true --sep-noisy-node true --train-loss-type smooth_l1_loss --addh true" > cmd.sh

# python -u scripts/train.py --conf examples/ET-MD17_FT-angle.yaml --job-id md17_aspirin_angle_noisy_seq_ang_20_cord_0.005 --dataset-arg aspirin --pretrained-model /share/project/sharefs-skfeng/pre-training-via-denoising/experiments/pretrain_models/ET-PCQM4MV2_var0.04_var2_com_re_md17/step=399999-epoch=8-val_loss=0.1626-test_loss=0.2945-train_per_step=0.1681.ckpt --dihedral-angle-noise-scale 20 --position-noise-scale 0.005 --composition true --sep-noisy-node true > md17_aspirin_angle_noisy_seq_ang_50_cord_0.005.log


# python finetune_on_md17_seq.py --pretrain_models /share/project/sharefs-skfeng/pre-training-via-denoising/experiments/pretrain_models/ET-PCQM4MV2_var0.04_var2_com_re_md17/step=399999-epoch=8-val_loss=0.1626-test_loss=0.2945-train_per_step=0.1681.ckpt --conf ET-MD17_FT-angle_9500.yaml --add_cmd "--dihedral-angle-noise-scale 20 --position-noise-scale 0.005 --composition true --sep-noisy-node true --train-loss-type smooth_l1_loss"

md17_task = {'aspirin', 'benzene', 'ethanol', 'malonaldehyde', 'naphthalene', 'salicylic_acid', 'toluene', 'uracil'}

# md17_task = {'aspirin', 'benzene', 'ethanol', 'malonaldehyde'}

# md17_task = {'naphthalene', 'salicylic_acid', 'toluene', 'uracil'}

# md17_task = {'benzene', 'ethanol', 'malonaldehyde', 'naphthalene', 'salicylic_acid', 'toluene', 'uracil'}

# md17_task = {'benzene', 'ethanol', 'naphthalene', 'toluene', 'uracil'}
# md17_task = {'uracil'}
# md17_task = {'malonaldehyde', 'salicylic_acid'}
# md17_task = {'benzene', 'toluene', 'uracil'}
# md17_task = {'malonaldehyde'}
md17_task = {'aspirin'}
md17_task = {'ethanol', 'malonaldehyde', 'naphthalene', 'aspirin'}
# md17_task = {'naphthalene'}
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    # parser.add_argument("--pretrain_model", type=str, default="sce")
    parser.add_argument('--pretrain_models', nargs='+', help='pretrain_models', required=False, default=None)
    parser.add_argument("--job_prefix", type=str, default="job_prefix")
    parser.add_argument("--config", type=str, default="ET-MD17.yaml")
    parser.add_argument("--add_cmd", type=str, default="")
    parser.add_argument("--seeds", type=int, default=3)

    args = parser.parse_args()
    # pretrain_model = args.pretrain_model
    pretrain_models = args.pretrain_models
    job_prefix = args.job_prefix
    config = args.config
    add_cmd = args.add_cmd

    seeds = args.seeds

    if len(add_cmd):
        job_info = '_'.join([ele.strip().strip('-') for ele in add_cmd.split()])
        job_info = job_info.replace('-', '_')
    else:
        job_info = ''
    
    gpu_id = 0
    if pretrain_models is not None:
        for pretrain_model in pretrain_models:
            job_prefix = os.path.basename(os.path.dirname(pretrain_model))
            for seed in range(1, seeds+1):
                for task in md17_task:
                    base_cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python -u scripts/train.py --conf examples/{config}  --job-id {job_prefix}_md17_{task}_{job_info}_finetuning_seed{seed} --dataset-arg {task} --pretrained-model {pretrain_model} {add_cmd} --seed {seed} > {job_prefix}_qm9_{task}_{job_info}_finetuning_seed{seed}.log 2>&1 &'
                    print(base_cmd + '\n\n')
                    gpu_id += 1
                    # os.system(base_cmd)
            pass
    else:
        # train from scrach
        for task in md17_task:
            # nohup rlaunch --cpu=16 --gpu=1 --memory=32000 --max-wait-time=36000 --   &
            base_cmd = f'python -u scripts/train.py --conf examples/{config}  --job-id {job_prefix}_md17_{task}_finetuning --dataset-arg {task} > {job_prefix}_qm9_{task}_finetuning.log'
            print(base_cmd)
            # os.system(base_cmd)