import torch
import os
gpu_nums = torch.cuda.device_count()
gpu_ava = torch.cuda.is_available()
gpu_info = f'gpu_numbers: {gpu_nums}, gpu is avai: {gpu_ava}\n'


test_cmd = 'nvidia-smi > nvidia-smi.txt'
os.system(test_cmd)

with open('gpu.info', 'w') as gw:
    gw.write(gpu_info)