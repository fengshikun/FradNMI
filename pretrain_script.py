import os


config_lst = ['examples/ET-PCQM4MV2_var0.1.yaml', 'examples/ET-PCQM4MV2_BIAS_1k_var0.1.yaml', 'examples/ET-PCQM4MV2_BIAS_1k_var0.1_violate.yaml']

config_lst = ['examples/ET-PCQM4MV2_BIAS_1k_var0.1.yaml', 'examples/ET-PCQM4MV2_BIAS_1k_var0.1_violate.yaml']

config_lst = ['examples/ET-PCQM4MV2_var0.1.yaml']


config_lst = ['examples/ET-PCQM4MV2_dih_var0.04_var1.yaml', 'examples/ET-PCQM4MV2_dih_var0.04_var1_compose.yaml', 'examples/ET-PCQM4MV2_dih_var0.04_var2.yaml', 'examples/ET-PCQM4MV2_dih_var0.04_var2_compose.yaml', 'examples/ET-PCQM4MV2_dih_var0.04_var20.yaml', 'examples/ET-PCQM4MV2_dih_var0.04_var20_compose.yaml',]

config_lst = ['examples/ET-PCQM4MV2_dih_var0.04_var1_no_norm.yaml']

config_lst = ['examples/ET-PCQM4MV2_dih_var0.04_var1_compose_no_norm.yaml', 'examples/ET-PCQM4MV2_dih_var0.04_var20_no_norm.yaml', 'examples/ET-PCQM4MV2_dih_var0.04_var20_compose_no_norm.yaml']

config_lst = ['examples/ET-PCQM4MV2_dih_var0.04_var1_compose_no_norm.yaml']

config_lst = ['examples/ET-PCQM4MV2_dih_var0.04_var1_both.yaml', 'examples/ET-PCQM4MV2_var0.04.yaml']

config_lst = ['examples/ET-PCQM4MV2_dih_var0.04_var0.1_compose.yaml', 'examples/ET-PCQM4MV2_dih_var0.04_var0.01_compose.yaml', 'examples/ET-PCQM4MV2_dih_var0.04_var0.1_com_re.yaml', 'examples/ET-PCQM4MV2_dih_var0.04_var0.1_re.yaml', 'examples/ET-PCQM4MV2_dih_var0.04_var1_com_re.yaml', 'examples/ET-PCQM4MV2_dih_var0.04_var1_re.yaml', 'examples/ET-PCQM4MV2_dih_var0.04_var2_com_re.yaml']

config_lst = ['examples/ET-PCQM4MV2_dih_var0.04_var0.01_com_re.yaml', 'examples/ET-PCQM4MV2_dih_var0.04_var0.01_re.yaml', 'examples/ET-PCQM4MV2_dih_var0.04_var0.1_com_re.yaml', 'examples/ET-PCQM4MV2_dih_var0.04_var0.1_re.yaml', 'examples/ET-PCQM4MV2_dih_var0.04_var1_com_re.yaml', 'examples/ET-PCQM4MV2_dih_var0.04_var1_re.yaml', 'examples/ET-PCQM4MV2_dih_var0.04_var2_com_re.yaml']


config_lst = ['examples/ET-PCQM4MV2_dih_var0.04_var0.01_re.yaml', 
'examples/ET-PCQM4MV2_dih_var0.04_var0.1_re.yaml']

# config_lst = ['examples/ET-PCQM4MV2_var0.4_s.yaml']
config_lst = ['examples/ET-PCQM4MV2_dih_var0.04_var5_com_re.yaml']

config_lst = ['examples/ET-PCQM4MV2_var0.4_s_md17.yaml', 'examples/ET-PCQM4MV2_var0.04_s_md17.yaml']


config_lst = ['examples/ET-PCQM4MV2_var0.04_var20_com_re_md17.yaml']

config_lst = ['examples/ET-PCQM4MV2_var0.04_var0.1_com_re_md17_decay.yaml', 'examples/ET-PCQM4MV2_var0.04_var2_com_re_md17_decay.yaml', 'examples/ET-PCQM4MV2_dih_var0.04_var2_com_re.yaml']

config_lst = ['examples/ET-PCQM4MV2_dih_var0.04_var1_com_re_40epoch.yaml', 'examples/ET-PCQM4MV2_dih_var0.04_var2_com_re_40epoch.yaml', 'examples/ET-PCQM4MV2_dih_var0.04_var3_com_re_40epoch.yaml', 'examples/ET-PCQM4MV2_dih_var0.04_var4_com_re_40epoch.yaml']

config_lst = ['examples/ET-PCQM4MV2_dih_var0.04_equilibrim_80epoch_addh.yaml', 'examples/ET-PCQM4MV2_dih_var0.04_equilibrim_eqw_80epoch_addh.yaml', 'examples/ET-PCQM4MV2_dih_var0.04_var1_com_re_40epoch_addh.yaml', 'examples/ET-PCQM4MV2_dih_var0.04_var2_com_re_40epoch_addh.yaml', 'examples/ET-PCQM4MV2_dih_var0.04_var3_com_re_40epoch_addh.yaml' ,'examples/ET-PCQM4MV2_dih_var0.04_var2_com_re_32epoch_addh_inte.yaml', 'examples/ET-PCQM4MV2_var0.04_var2_com_re_md17_32epoch_addh_inte.yaml']


config_lst = ['examples/ET-PCQM4MV2_var0.04_var1_com_re_md17.yaml', 'examples/ET-PCQM4MV2_var0.4_var2_com_re_md17.yaml', 'examples/ET-PCQM4MV2_var0.04_var2_com_re_md17.yaml', 'examples/ET-PCQM4MV2_var0.004_var2_com_re_md17.yaml', 'examples/ET-PCQM4MV2_var0.04_var20_com_re_md17.yaml', 'examples/ET-PCQM4MV2_dih_var0.4_var2_com_re.yaml', 'examples/ET-PCQM4MV2_dih_var0.004_var2_com_re.yaml']

for config in config_lst:
    job_name = os.path.split(config)[-1][:-5]
    #  + '_denan'
    # base_cmd = f'nohup rlaunch --cpu=8 --gpu=1 --memory=128000 --max-wait-time=36000 -- python scripts/train.py --conf {config} --layernorm-on-vec whitened --job-id {job_name} > {job_name}.log &'
    base_cmd = f'python scripts/train.py --conf {config} --layernorm-on-vec whitened --job-id {job_name} > {job_name}.log'
    print(base_cmd + '\n\n')
    # os.system(base_cmd)