import os
import argparse
import re
import glob

task_dict = {'dipole_moment': 0, 'isotropic_polarizability': 1, 'homo': 2, 'lumo': 3, 'gap': 4, 'electronic_spatial_extent': 5, 'zpve': 6, 'energy_U0': 7, 'energy_U': 8, 'enthalpy_H': 9, 'free_energy': 10, 'heat_capacity': 11}

exp_dir_prefix = '/home/AI4Science/fengsk/Pretraining-Denoising/experiments/'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument("--job_prefix", type=str, default="job_prefix")
    args = parser.parse_args()
    job_prefix = args.job_prefix
    # job_prefix = 'pretrain_baseline_var0.04'

    for task in task_dict:
        log_file = f'{job_prefix}_qm9_{task}_finetuning.log'
        exp_dir = f'{exp_dir_prefix}{job_prefix}_qm9_{task}_finetuning'
        with open(log_file, 'r') as lr:
            ckpt_file = glob.glob(f'{exp_dir}/*.ckpt')
            ckpt_file.remove(os.path.join(os.path.dirname(ckpt_file[-1]), 'last.ckpt'))
            ckpt_file.sort(key=lambda cf: int(os.path.split(cf)[-1].split('-')[0].split('=')[1]))
            file_name = os.path.split(ckpt_file[-1])[-1]
            pat = r'\d+\.\d+|\d+'
            result = re.findall(pat, file_name)
            if 'loss=nan' in lr.read():
                assert result[1] != '339'
                print(f'{task}: {result[-2]}(epoch {result[1]})')
            else:
                if result[1] != '339':
                    print(f'{task}: {result[-2]} not finished, current epoch {result[1]}')
                else:
                    print(f'{task}: {result[-2]}')
                # pass