import os
import argparse


#  python finetune_on_md17_seq.py --pretrain_models /share/project/sharefs-skfeng/pre-training-via-denoising/experiments/pretrain_models/ET-PCQM4MV2_var0.04_var2_com_re_md17/step=399999-epoch=8-val_loss=0.1626-test_loss=0.2945-train_per_step=0.1681.ckpt --config ET-MD17_FT-angle.yaml --add_cmd="--dihedral-angle-noise-scale 20 --position-noise-scale 0.005 --composition true --sep-noisy-node true --train-loss-type smooth_l1_loss --addh true" > cmd.sh

# python -u scripts/train.py --conf examples/ET-MD17_FT-angle.yaml --job-id md17_aspirin_angle_noisy_seq_ang_20_cord_0.005 --dataset-arg aspirin --pretrained-model /share/project/sharefs-skfeng/pre-training-via-denoising/experiments/pretrain_models/ET-PCQM4MV2_var0.04_var2_com_re_md17/step=399999-epoch=8-val_loss=0.1626-test_loss=0.2945-train_per_step=0.1681.ckpt --dihedral-angle-noise-scale 20 --position-noise-scale 0.005 --composition true --sep-noisy-node true > md17_aspirin_angle_noisy_seq_ang_50_cord_0.005.log

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
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    # parser.add_argument("--pretrain_model", type=str, default="sce")
    parser.add_argument('--pretrain_models', nargs='+', help='pretrain_models', required=False, default=None)
    parser.add_argument("--job_prefix", type=str, default="job_prefix")
    parser.add_argument("--config", type=str, default="ET-MD17.yaml")
    parser.add_argument("--add_cmd", type=str, default="")
    args = parser.parse_args()
    # pretrain_model = args.pretrain_model
    pretrain_models = args.pretrain_models
    job_prefix = args.job_prefix
    config = args.config
    add_cmd = args.add_cmd

    if len(add_cmd):
        job_info = '_'.join([ele.strip().strip('-') for ele in add_cmd.split()])
        job_info = job_info.replace('-', '_')
    else:
        job_info = ''
    
    gpu_id = 0
    if pretrain_models is not None:
        for pretrain_model in pretrain_models:
            job_prefix = os.path.basename(os.path.dirname(pretrain_model))
            for task in md17_task:
                base_cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python -u scripts/train.py --conf examples/{config}  --job-id {job_prefix}_md17_{task}_{job_info}_finetuning --dataset-arg {task} --pretrained-model {pretrain_model} {add_cmd} > {job_prefix}_qm9_{task}_{job_info}_finetuning.log 2>&1 &'
                print(base_cmd + '\n\n')
                gpu_id += 1
                # os.system(base_cmd)
            pass
    else:
        # train from scrach
        for task in md17_task:
            base_cmd = f'nohup rlaunch --cpu=16 --gpu=1 --memory=32000 --max-wait-time=36000 --  python -u scripts/train.py --conf examples/{config}  --job-id {job_prefix}_md17_{task}_finetuning --dataset-arg {task} > {job_prefix}_qm9_{task}_finetuning.log &'
            print(base_cmd)
            # os.system(base_cmd)