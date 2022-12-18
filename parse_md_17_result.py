import os
import argparse
import re
import glob

task_dict = {'aspirin', 'benzene', 'ethanol', 'malonaldehyde', 'naphthalene', 'salicylic_acid', 'toluene', 'uracil'}

dy_pat = 'test_loss_dy(.*)[-+]?[0-9]*\.?[0-9]+'
y_pat = 'test_loss_y(.*)[-+]?[0-9]*\.?[0-9]+'



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument("--job_prefix", type=str, default="job_prefix")
    args = parser.parse_args()
    job_prefix = args.job_prefix
    # job_prefix = 'pretrain_baseline_var0.04_BIAS_violate_2'

    for task in task_dict:
        log_file = f'{job_prefix}_qm9_{task}_finetuning.log'
        if not os.path.exists(log_file):
            print(f'task {log_file} not found')
            continue
        with open(log_file, 'r') as lr:
            content = lr.read()
            # content = '+++++++++Best res is: 0.8202898550724638++++++++++++'
            match_res = re.findall(y_pat, content)
            dmatch_res = re.findall(dy_pat, content)
            if not len(dmatch_res) or not len(match_res):
                print(f'{task} is invalid')
                continue
            print(f'{task} energy and force is {match_res[0]} {dmatch_res[0]}')
            # num_res = []
            # for mres in match_res:
            #     num_res.append(float(mres[2]))
            # if len(num_res) == 3:
            #     print(f"{data_set}: {np.mean(num_res):.3f}({np.std(num_res):.3f})")
            # else:
            #     print(f"parse {data_set} error")
            
            # pass