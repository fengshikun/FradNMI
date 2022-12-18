import os
import argparse






md17_task = {'aspirin', 'benzene', 'ethanol', 'malonaldehyde', 'naphthalene', 'salicylic_acid', 'toluene', 'uracil'}

md17_task = {'aspirin', 'benzene', 'ethanol', 'malonaldehyde'}

md17_task = {'naphthalene', 'salicylic_acid', 'toluene', 'uracil'}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    # parser.add_argument("--pretrain_model", type=str, default="sce")
    parser.add_argument('--pretrain_models', nargs='+', help='pretrain_models', required=False, default=None)
    parser.add_argument("--job_prefix", type=str, default="job_prefix")
    parser.add_argument("--config", type=str, default="ET-MD17.yaml")
    args = parser.parse_args()
    # pretrain_model = args.pretrain_model
    pretrain_models = args.pretrain_models
    job_prefix = args.job_prefix
    config = args.config
    
    if pretrain_models is not None:
        for pretrain_model in pretrain_models:
            job_prefix = os.path.basename(os.path.dirname(pretrain_model))
            for task in md17_task:
                base_cmd = f'nohup rlaunch --cpu=16 --gpu=1 --memory=32000 --max-wait-time=36000 --  python -u scripts/train.py --conf examples/{config}  --job-id {job_prefix}_md17_{task}_finetuning --dataset-arg {task} --pretrained-model {pretrain_model} > {job_prefix}_qm9_{task}_finetuning.log &'
                print(base_cmd)
                os.system(base_cmd)
            pass
    else:
        # train from scrach
        for task in md17_task:
            base_cmd = f'nohup rlaunch --cpu=16 --gpu=1 --memory=32000 --max-wait-time=36000 --  python -u scripts/train.py --conf examples/{config}  --job-id {job_prefix}_md17_{task}_finetuning --dataset-arg {task} > {job_prefix}_qm9_{task}_finetuning.log &'
            # print(base_cmd)
            # os.system(base_cmd)