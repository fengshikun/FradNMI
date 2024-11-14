# denoise angle pretraining(bat denoise, rotation denoise) 2 GPU

# baseline
# CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py --conf examples/ET-PCQM4MV2_dih_var0.04_var2_com_re.yaml --layernorm-on-vec whitened --job-id frad_pretraining --num-epochs 8 > frad_pretraining.log 2>&1 & 
# 

# denoise angle

CUDA_VISIBLE_DEVICES=4 python -u scripts/train.py --conf examples/ET-PCQM4MV2_dih_var0.04_var2_com.yaml --layernorm-on-vec whitened --job-id frad_pretraining_denoise_angle --num-epochs 8 --dataset-root /data/protein/SKData/DenoisingData/pcq > frad_pretraining_denoise_angle.log 2>&1 & # denoise angle

CUDA_VISIBLE_DEVICES=5 python -u scripts/train.py --conf examples/ET-PCQM4MV2_dih_var0.04_var2_com.yaml --layernorm-on-vec whitened --job-id frad_pretraining_denoise_angle_bat --num-epochs 8 --bat-noise true --dataset-root /data/protein/SKData/DenoisingData/pcq > frad_pretraining_denoise_angle_bat.log 2>&1 &



# painn
CUDA_VISIBLE_DEVICES=6 python -u scripts/train.py --conf examples/ET-PCQM4MV2_dih_var0.04_var2_com_re.yaml --layernorm-on-vec whitened --job-id frad_pretraining_painn --num-epochs 8 --model painn --embedding-dimension 128 --dataset-root /data/protein/SKData/DenoisingData/pcq  > frad_pretraining_painn.log 2>&1 & 


CUDA_VISIBLE_DEVICES=7 python -u scripts/train.py --conf examples/ET-PCQM4MV2_dih_var0.04_var2_com_re.yaml --layernorm-on-vec whitened --job-id frad_pretraining_egnn --num-epochs 8 --model egnn --dataset-root /data/protein/SKData/DenoisingData/pcq > frad_pretraining_egnn.log 2>&1 & 




CUDA_VISIBLE_DEVICES=7 python -u scripts/train.py --conf examples/ET-PCQM4MV2_dih_var0.04_var2_com_re.yaml --layernorm-on-vec whitened --job-id frad_pretraining_egnn_h128_n10 --num-epochs 10 --model egnn --dataset-root /data/protein/SKData/DenoisingData/pcq  --hidden-nf 128 --n-layers 10 > frad_pretraining_egnn_h128_n10.log 2>&1 & 


CUDA_VISIBLE_DEVICES=7 python -u scripts/train.py --conf examples/ET-PCQM4MV2_dih_var0.04_var2_com_re.yaml --layernorm-on-vec whitened --job-id frad_pretraining_egnn2 --num-epochs 8 --model egnn --dataset-root /data/protein/SKData/DenoisingData/pcq > frad_pretraining_egnn2.log 2>&1 & 

# pretraining for 10w
CUDA_VISIBLE_DEVICES=3 python -u scripts/train.py --conf examples/ET-PCQM4MV2_dih_var0.04_var2_com_re.yaml --layernorm-on-vec whitened --job-id frad_pretraining_rdkit_10w --num-epochs 8 --dataset PCQM4MV2_Dihedral3  --dataset-root /data/protein/SKData/DenoisingData/pcq  > frad_pretraining_rdkit_10w.log 2>&1 & 


CUDA_VISIBLE_DEVICES=3 python -u scripts/train.py --conf examples/ET-PCQM4MV2_dih_var0.04_var2_com_re.yaml --layernorm-on-vec whitened --job-id frad_pretraining_10w --num-epochs 8 --dataset PCQM4MV2_Dihedral4  --dataset-root /data/protein/SKData/DenoisingData/pcq  > frad_pretraining_10w.log 2>&1 & 






### finetuneing qm9 of frad 10w
CUDA_VISIBLE_DEVICES=3 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id frad_pretraining_10w_lumo --dataset-arg lumo --pretrained-model /data/protein/SKData/Frad_NMI/FradNMI/experiments/frad_pretraining_10w/step=11407-epoch=7-val_loss=0.2065-test_loss=0.2143-train_per_step=0.2014.ckpt --denoising-weight 0.1  --dataset-root /data/protein/SKData/DenoisingData/qm9 > frad_pretraining_10w_lumo.log 2>&1 &

CUDA_VISIBLE_DEVICES=6 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id frad_pretraining_10w_homo --dataset-arg homo --pretrained-model /data/protein/SKData/Frad_NMI/FradNMI/experiments/frad_pretraining_10w/step=11407-epoch=7-val_loss=0.2065-test_loss=0.2143-train_per_step=0.2014.ckpt --denoising-weight 0.1  --dataset-root /data/protein/SKData/DenoisingData/qm9 > frad_pretraining_10w_homo.log 2>&1 &

CUDA_VISIBLE_DEVICES=7 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id frad_pretraining_10w_gap --dataset-arg gap --pretrained-model /data/protein/SKData/Frad_NMI/FradNMI/experiments/frad_pretraining_10w/step=11407-epoch=7-val_loss=0.2065-test_loss=0.2143-train_per_step=0.2014.ckpt --train-loss-type smooth_l1_loss  --dataset-root /data/protein/SKData/DenoisingData/qm9 > frad_pretraining_10w_gap.log 2>&1 &


CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id frad_pretraining_rdkit_10w_lumo --dataset-arg lumo --pretrained-model /data/protein/SKData/Frad_NMI/FradNMI/experiments/frad_pretraining_rdkit_10w/step=11407-epoch=7-val_loss=0.2023-test_loss=0.2045-train_per_step=0.1883.ckpt --denoising-weight 0.1  --dataset-root /data/protein/SKData/DenoisingData/qm9 > frad_pretraining_rdkit_10w_lumo.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id frad_pretraining_rdkit_10w_homo --dataset-arg homo --pretrained-model /data/protein/SKData/Frad_NMI/FradNMI/experiments/frad_pretraining_rdkit_10w/step=11407-epoch=7-val_loss=0.2023-test_loss=0.2045-train_per_step=0.1883.ckpt --denoising-weight 0.1  --dataset-root /data/protein/SKData/DenoisingData/qm9 > frad_pretraining_rdkit_10w_homo.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id frad_pretraining_rdkit_10w_gap --dataset-arg gap --pretrained-model /data/protein/SKData/Frad_NMI/FradNMI/experiments/frad_pretraining_rdkit_10w/step=11407-epoch=7-val_loss=0.2023-test_loss=0.2045-train_per_step=0.1883.ckpt --train-loss-type smooth_l1_loss  --dataset-root /data/protein/SKData/DenoisingData/qm9 > frad_pretraining_rdkit_10w_gap.log 2>&1 &


CUDA_VISIBLE_DEVICES=4 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id frad_pretraining_denoise_angle_lumo --dataset-arg lumo --pretrained-model /data/protein/SKData/Frad_NMI/FradNMI/experiments/frad_pretraining_denoise_angle/step=386103-epoch=7-val_loss=0.2156-test_loss=0.2038-train_per_step=0.1778.ckpt --denoising-weight 0.1  --dataset-root /data/protein/SKData/DenoisingData/qm9 > frad_pretraining_denoise_angle_lumo.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id frad_pretraining_denoise_angle_homo --dataset-arg homo --pretrained-model /data/protein/SKData/Frad_NMI/FradNMI/experiments/frad_pretraining_denoise_angle/step=386103-epoch=7-val_loss=0.2156-test_loss=0.2038-train_per_step=0.1778.ckpt --denoising-weight 0.1  --dataset-root /data/protein/SKData/DenoisingData/qm9 > frad_pretraining_denoise_angle_homo.log 2>&1 &

CUDA_VISIBLE_DEVICES=6 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id frad_pretraining_denoise_angle_gap --dataset-arg gap --pretrained-model /data/protein/SKData/Frad_NMI/FradNMI/experiments/frad_pretraining_denoise_angle/step=386103-epoch=7-val_loss=0.2156-test_loss=0.2038-train_per_step=0.1778.ckpt --train-loss-type smooth_l1_loss  --dataset-root /data/protein/SKData/DenoisingData/qm9 > frad_pretraining_denoise_angle_gap.log 2>&1 &


# pretraining on qm9
CUDA_VISIBLE_DEVICES=6 python -u scripts/train.py --conf examples/ET-QM9_dih_var0.04_var2_com_re.yaml --layernorm-on-vec whitened --job-id frad_pretraining_qm9_10w --num-epochs 8  --dataset-root /data/protein/SKData/DenoisingData/qm9 --dataset-arg lumo   > frad_pretraining_qm9_10w.log 2>&1 & 