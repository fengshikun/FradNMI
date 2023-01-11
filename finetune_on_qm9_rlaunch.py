import os
import argparse





qm9_task = {'dipole_moment': 0, 'isotropic_polarizability': 1, 'homo': 2, 'lumo': 3, 'gap': 4, 'electronic_spatial_extent': 5, 'zpve': 6, 'energy_U0': 7, 'energy_U': 8, 'enthalpy_H': 9, 'free_energy': 10, 'heat_capacity': 11}

qm9_task = {'homo': 2, 'lumo': 3, 'gap': 4}

qm9_task = {'electronic_spatial_extent': 5, 'zpve': 6}

# qm9_task = {'homo': 2}

# qm9_task = {'dipole_moment': 0, 'isotropic_polarizability': 1, 'electronic_spatial_extent': 5}
# qm9_task = {'gap': 4}
qm9_task = {'dipole_moment': 0, 'isotropic_polarizability': 1, 'electronic_spatial_extent': 5, 'zpve': 6}

qm9_task = {'electronic_spatial_extent': 5, 'zpve': 6}

qm9_task = {'isotropic_polarizability': 1}

qm9_task = {'energy_U0': 7, 'energy_U': 8, 'enthalpy_H': 9, 'free_energy': 10, 'heat_capacity': 11}


qm9_task = {'dipole_moment': 0, 'isotropic_polarizability': 1, 'electronic_spatial_extent': 5, 'zpve': 6, 'energy_U0': 7, 'energy_U': 8, 'enthalpy_H': 9, 'free_energy': 10, 'heat_capacity': 11}

# qm9_task = {'energy_U0': 8, 'enthalpy_H': 9, 'free_energy': 11}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    # parser.add_argument("--pretrain_model", type=str, default="sce")
    parser.add_argument('--pretrain_models', nargs='+', help='pretrain_models', required=False, default=None)
    parser.add_argument("--job_prefix", type=str, default="job_prefix")
    parser.add_argument("--config", type=str, default="ET-QM9-FT.yaml")
    args = parser.parse_args()
    # pretrain_model = args.pretrain_model
    pretrain_models = args.pretrain_models
    job_prefix = args.job_prefix
    config = args.config

    for pretrain_model in pretrain_models:
        job_prefix = os.path.basename(os.path.dirname(pretrain_model))
        for task in qm9_task:
            if task in ['energy_U0', 'energy_U', 'enthalpy_H', 'free_energy']:
                config = 'ET-QM9-FT-nt.yaml'
            # base_cmd = f'srun -mem=64000 --gres=gpu:1 --time 6-12:00:00 --job-name {job_prefix}_qm9_{task}_finetuning python -u scripts/train.py  scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id {job_prefix}_qm9_{task}_finetuning --dataset-arg {task} --pretrained-model {pretrain_model} > {job_prefix}_qm9_{task}_finetuning.log 2>&1 &'
            base_cmd = f'nohup rlaunch --cpu=16 --gpu=1 --memory=32000 --max-wait-time=36000 --  python -u scripts/train.py --conf examples/{config} --layernorm-on-vec whitened --job-id {job_prefix}_qm9_{task}_finetuning --dataset-arg {task} --pretrained-model {pretrain_model} > {job_prefix}_qm9_{task}_finetuning.log &'
            print(base_cmd)
            # os.system(base_cmd)
        pass