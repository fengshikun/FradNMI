python -u scripts/train.py --conf examples/ET-PCQM4MV2_dih_var0.04_var2_com_re.yaml --layernorm-on-vec whitened --job-id frad_1w --num-epochs 8 --train-size 10000 --batch-size 8 --lr-warmup-steps 100 --lr-cosine-length 10000 --num-steps 10000;
python -u scripts/train.py --conf examples/ET-PCQM4MV2_dih_var0.04_var2_com_re.yaml --layernorm-on-vec whitened --job-id frad_10w --num-epochs 8 --train-size 100000 --batch-size 16 --lr-warmup-steps 1000 --lr-cosine-length 50000 --num-steps 50000;
python -u scripts/train.py --conf examples/ET-PCQM4MV2_dih_var0.04_var2_com_re.yaml --layernorm-on-vec whitened --job-id frad_20w --num-epochs 8 --train-size 200000 --batch-size 16 --lr-warmup-steps 1000 --lr-cosine-length 100000 --num-steps 100000;
python -u scripts/train.py --conf examples/ET-PCQM4MV2_dih_var0.04_var2_com_re.yaml --layernorm-on-vec whitened --job-id frad_50w --num-epochs 8 --train-size 400000 --batch-size 32 --lr-warmup-steps 1000 --lr-cosine-length 100000 --num-steps 100000;


python -u scripts/train.py --conf examples/ET-PCQM4MV2_dih_var0.04_var2_com_re.yaml --layernorm-on-vec whitened --job-id frad_80w --num-epochs 8 --train-size 800000 --batch-size 64 --lr-warmup-steps 10000 --lr-cosine-length 100000 --num-steps 100000;
python -u scripts/train.py --conf examples/ET-PCQM4MV2_dih_var0.04_var2_com_re.yaml --layernorm-on-vec whitened --job-id frad_100w --num-epochs 8 --train-size 1000000 --batch-size 64 --lr-warmup-steps 10000 --lr-cosine-length 125000 --num-steps 125000;
python -u scripts/train.py --conf examples/ET-PCQM4MV2_dih_var0.04_var2_com_re.yaml --layernorm-on-vec whitened --job-id frad_150w --num-epochs 8 --train-size 1600000 --batch-size 64 --lr-warmup-steps 10000 --lr-cosine-length 200000 --num-steps 200000;
python -u scripts/train.py --conf examples/ET-PCQM4MV2_dih_var0.04_var2_com_re.yaml --layernorm-on-vec whitened --job-id frad_200w --num-epochs 8 --train-size 2000000 --batch-size 64 --lr-warmup-steps 10000 --lr-cosine-length 250000 --num-steps 250000;
python -u scripts/train.py --conf examples/ET-PCQM4MV2_dih_var0.04_var2_com_re.yaml --layernorm-on-vec whitened --job-id frad_250w --num-epochs 8 --train-size 2500000 --batch-size 64 --lr-warmup-steps 10000 --lr-cosine-length 312500 --num-steps 312500;


# test the model

python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id frad_1w_homo --dataset-arg homo --pretrained-model /home/fengshikun/Pretraining-DenoisingNMI/experiments/frad_1w/step=9999-epoch=7-val_loss=0.2453-test_loss=0.2351-train_per_step=0.2949.ckpt --denoising-weight 0.1 --bond-length-scale 0.0 


python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id frad_10w_homo --dataset-arg homo --pretrained-model /home/fengshikun/Pretraining-DenoisingNMI/experiments/frad_10w/step=49999-epoch=7-val_loss=0.1807-test_loss=0.1723-train_per_step=0.1831.ckpt --denoising-weight 0.1 --bond-length-scale 0.0 

python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id frad_20w_homo --dataset-arg homo --pretrained-model /home/fengshikun/Pretraining-DenoisingNMI/experiments/frad_20w/step=99999-epoch=7-val_loss=0.1674-test_loss=0.1572-train_per_step=0.1507.ckpt --denoising-weight 0.1 --bond-length-scale 0.0 


python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id frad_50w_homo --dataset-arg homo --pretrained-model /home/fengshikun/Pretraining-DenoisingNMI/experiments/frad_50w/step=99999-epoch=7-val_loss=0.1537-test_loss=0.1620-train_per_step=0.1730.ckpt --denoising-weight 0.1 --bond-length-scale 0.0 


python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id frad_80w_homo --dataset-arg homo --pretrained-model /home/fengshikun/Pretraining-DenoisingNMI/experiments/frad_80w/step=99999-epoch=7-val_loss=0.1592-test_loss=0.1544-train_per_step=0.1481.ckpt --denoising-weight 0.1 --bond-length-scale 0.0 




python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id frad_100w_homo --dataset-arg homo --pretrained-model /home/fengshikun/Pretraining-DenoisingNMI/experiments/frad_100w/step=124999-epoch=7-val_loss=0.1630-test_loss=0.1431-train_per_step=0.1574.ckpt --denoising-weight 0.1 --bond-length-scale 0.0 



python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id frad_150w_homo --dataset-arg homo --pretrained-model /home/fengshikun/Pretraining-DenoisingNMI/experiments/frad_150w/step=199999-epoch=7-val_loss=0.1472-test_loss=0.1479-train_per_step=0.1425.ckpt --denoising-weight 0.1 --bond-length-scale 0.0 


python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id frad_200w_homo --dataset-arg homo --pretrained-model /home/fengshikun/Pretraining-DenoisingNMI/experiments/frad_200w/step=249999-epoch=7-val_loss=0.1525-test_loss=0.1384-train_per_step=0.1574.ckpt --denoising-weight 0.1 --bond-length-scale 0.0 






====================


python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id frad_1w_lumo --dataset-arg lumo --pretrained-model /home/fengshikun/Pretraining-DenoisingNMI/experiments/frad_1w/step=9999-epoch=7-val_loss=0.2453-test_loss=0.2351-train_per_step=0.2949.ckpt --denoising-weight 0.1 --bond-length-scale 0.0




python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id frad_1w_gap --dataset-arg gap --pretrained-model /home/fengshikun/Pretraining-DenoisingNMI/experiments/frad_1w/step=9999-epoch=7-val_loss=0.2453-test_loss=0.2351-train_per_step=0.2949.ckpt --train-loss-type smooth_l1_loss --bond-length-scale 0.0



# test md17 command


# bash /share/project/quhx/nv.sh; export LD_LIBRARY_PATH=/usr/local/cuda-11/lib64:$LD_LIBRARY_PATH;
cd /home/fengshikun/Pretraining-DenoisingNMI; 

python -u scripts/train.py --conf examples/ET-MD17_FT-angle_9500.yaml --job-id ET-PCQM4MV2_var0.04_var2_com_re_md17_md17_naphthalene_dihedral_angle_noise_scale_20_position_noise_scale_0.005_composition_true_sep_noisy_node_true_train_loss_type_smooth_l1_loss_finetuning --dataset-arg naphthalene --pretrained-model /home/fengshikun/Pretraining-Denoising-models/experiments/ET-PCQM4MV2_var0.04_var2_com_re_md17/step=399999-epoch=8-val_loss=0.1626-test_loss=0.2945-train_per_step=0.1681.ckpt --dihedral-angle-noise-scale 20 --position-noise-scale 0.005 --composition true --sep-noisy-node true --train-loss-type smooth_l1_loss > ET-PCQM4MV2_var0.04_var2_com_re_md17_qm9_naphthalene_dihedral_angle_noise_scale_20_position_noise_scale_0.005_composition_true_sep_noisy_node_true_train_loss_type_smooth_l1_loss_finetuning.log
