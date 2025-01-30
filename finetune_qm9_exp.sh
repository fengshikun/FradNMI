CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id egnn_wop_lumo --dataset-arg lumo   --denoising-weight 0.1  --model egnn --dataset-root /nfs/SKData/DenoisingData/qm9 > egnn_wop_lumo.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id egnn_wop_homo --dataset-arg homo   --denoising-weight 0.1  --model egnn --dataset-root /nfs/SKData/DenoisingData/qm9 > egnn_wop_homo.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id egnn_wop_gap --dataset-arg gap   --train-loss-type smooth_l1_loss  --model egnn --dataset-root /nfs/SKData/DenoisingData/qm9 > egnn_wop_gap.log 2>&1 &


# wo nn

CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id egnn_wop_lumo_wnn --dataset-arg lumo   --denoising-weight 0.0  --model egnn --dataset-root /nfs/SKData/DenoisingData/qm9 > egnn_wop_lumo_wnn.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id egnn_wop_homo_wnn --dataset-arg homo   --denoising-weight 0.0  --model egnn --dataset-root /nfs/SKData/DenoisingData/qm9 > egnn_wop_homo_wnn.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id egnn_wop_gap_wnn --dataset-arg gap  --denoising-weight 0.0  --train-loss-type smooth_l1_loss  --model egnn --dataset-root /nfs/SKData/DenoisingData/qm9 > egnn_wop_gap_wnn.log 2>&1 &



CUDA_VISIBLE_DEVICES=3 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id painn_wop_lumo --dataset-arg lumo  --denoising-weight 0.1  --model painn --embedding-dimension 128 --dataset-root /nfs/SKData/DenoisingData/qm9 > painn_wop_lumo.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id painn_wop_homo --dataset-arg homo  --denoising-weight 0.1  --model painn --embedding-dimension 128 --dataset-root /nfs/SKData/DenoisingData/qm9 > painn_wop_homo.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id painn_wop_gap --dataset-arg gap   --train-loss-type smooth_l1_loss  --model painn --embedding-dimension 128 --dataset-root /nfs/SKData/DenoisingData/qm9 > painn_wop_gap.log 2>&1 &

# without noisy nodes

CUDA_VISIBLE_DEVICES=3 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id painn_wop_lumo_wnn --dataset-arg lumo  --denoising-weight 0.0  --model painn --embedding-dimension 128 --dataset-root /nfs/SKData/DenoisingData/qm9 > painn_wop_lumo_wnn.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id painn_wop_homo_wnn --dataset-arg homo  --denoising-weight 0.0  --model painn --embedding-dimension 128 --dataset-root /nfs/SKData/DenoisingData/qm9 > painn_wop_homo_wnn.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id painn_wop_gap_wnn --dataset-arg gap --denoising-weight 0.0   --train-loss-type smooth_l1_loss  --model painn --embedding-dimension 128 --dataset-root /nfs/SKData/DenoisingData/qm9 > painn_wop_gap_wnn.log 2>&1 &


# pretraining
CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id painn_frad_lumo --dataset-arg lumo --pretrained-model /mnt/nfs-ssd/data/fengshikun/FradNMI/experiments/frad_pretraining_painn/step=386103-epoch=7-val_loss=0.2104-test_loss=0.2037-train_per_step=0.1890.ckpt --denoising-weight 0.1  --model painn --embedding-dimension 128 --dataset-root /nfs/SKData/DenoisingData/qm9 > painn_frad_lumo.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id painn_frad_homo --dataset-arg homo --pretrained-model /mnt/nfs-ssd/data/fengshikun/FradNMI/experiments/frad_pretraining_painn/step=386103-epoch=7-val_loss=0.2104-test_loss=0.2037-train_per_step=0.1890.ckpt --denoising-weight 0.1  --model painn --embedding-dimension 128 --dataset-root /nfs/SKData/DenoisingData/qm9 > painn_frad_homo.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id painn_frad_gap --dataset-arg gap --pretrained-model /mnt/nfs-ssd/data/fengshikun/FradNMI/experiments/frad_pretraining_painn/step=386103-epoch=7-val_loss=0.2104-test_loss=0.2037-train_per_step=0.1890.ckpt --train-loss-type smooth_l1_loss  --model painn --embedding-dimension 128 --dataset-root /nfs/SKData/DenoisingData/qm9 > painn_frad_gap.log 2>&1 &


CUDA_VISIBLE_DEVICES=3 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id egnn_frad_lumo --dataset-arg lumo --pretrained-model /mnt/nfs-ssd/data/fengshikun/FradNMI/experiments/frad_pretraining_egnn/step=386103-epoch=7-val_loss=0.2119-test_loss=0.2041-train_per_step=0.1908.ckpt --denoising-weight 0.1  --model egnn --dataset-root /nfs/SKData/DenoisingData/qm9  > egnn_frad_lumo.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id egnn_frad_homo --dataset-arg homo --pretrained-model /mnt/nfs-ssd/data/fengshikun/FradNMI/experiments/frad_pretraining_egnn/step=386103-epoch=7-val_loss=0.2119-test_loss=0.2041-train_per_step=0.1908.ckpt --denoising-weight 0.1  --model egnn --dataset-root /nfs/SKData/DenoisingData/qm9  > egnn_frad_homo.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id egnn_frad_gap --dataset-arg gap --pretrained-model /mnt/nfs-ssd/data/fengshikun/FradNMI/experiments/frad_pretraining_egnn/step=386103-epoch=7-val_loss=0.2119-test_loss=0.2041-train_per_step=0.1908.ckpt --train-loss-type smooth_l1_loss  --model egnn --dataset-root /nfs/SKData/DenoisingData/qm9  > egnn_frad_gap.log 2>&1 &




# pretraining without nn

CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id painn_frad_lumo_wonn --dataset-arg lumo --pretrained-model /mnt/nfs-ssd/data/fengshikun/FradNMI/experiments/frad_pretraining_painn/step=386103-epoch=7-val_loss=0.2104-test_loss=0.2037-train_per_step=0.1890.ckpt --denoising-weight 0.0  --model painn --embedding-dimension 128 --dataset-root /nfs/SKData/DenoisingData/qm9 > painn_frad_lumo_wonn.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id painn_frad_homo_wonn --dataset-arg homo --pretrained-model /mnt/nfs-ssd/data/fengshikun/FradNMI/experiments/frad_pretraining_painn/step=386103-epoch=7-val_loss=0.2104-test_loss=0.2037-train_per_step=0.1890.ckpt --denoising-weight 0.0  --model painn --embedding-dimension 128 --dataset-root /nfs/SKData/DenoisingData/qm9 > painn_frad_homo_wonn.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id painn_frad_gap_wonn --dataset-arg gap --pretrained-model /mnt/nfs-ssd/data/fengshikun/FradNMI/experiments/frad_pretraining_painn/step=386103-epoch=7-val_loss=0.2104-test_loss=0.2037-train_per_step=0.1890.ckpt --train-loss-type smooth_l1_loss  --model painn --embedding-dimension 128 --dataset-root /nfs/SKData/DenoisingData/qm9 --denoising-weight 0.0 > painn_frad_gap_wonn.log 2>&1 &


CUDA_VISIBLE_DEVICES=3 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id egnn_frad_lumo_wonn --dataset-arg lumo --pretrained-model /mnt/nfs-ssd/data/fengshikun/FradNMI/experiments/frad_pretraining_egnn/step=386103-epoch=7-val_loss=0.2119-test_loss=0.2041-train_per_step=0.1908.ckpt --denoising-weight 0.0  --model egnn --dataset-root /nfs/SKData/DenoisingData/qm9  > egnn_frad_lumo_wonn.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id egnn_frad_homo_wonn --dataset-arg homo --pretrained-model /mnt/nfs-ssd/data/fengshikun/FradNMI/experiments/frad_pretraining_egnn/step=386103-epoch=7-val_loss=0.2119-test_loss=0.2041-train_per_step=0.1908.ckpt --denoising-weight 0.0  --model egnn --dataset-root /nfs/SKData/DenoisingData/qm9  > egnn_frad_homo_wonn.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id egnn_frad_gap_wonn --dataset-arg gap --pretrained-model /mnt/nfs-ssd/data/fengshikun/FradNMI/experiments/frad_pretraining_egnn/step=386103-epoch=7-val_loss=0.2119-test_loss=0.2041-train_per_step=0.1908.ckpt --train-loss-type smooth_l1_loss  --model egnn --dataset-root /nfs/SKData/DenoisingData/qm9  --denoising-weight 0.0 > egnn_frad_gap_wonn.log 2>&1 &






# net abi egnn frad TODO pretrained model

CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id frad_egnn_h128_n10_lumo --dataset-arg lumo --pretrained-model /mnt/nfs-ssd/data/fengshikun/FradNMI/experiments/frad_pretraining_egnn_h128_n10/step=399999-epoch=7-val_loss=0.9758-test_loss=0.9907-train_per_step=0.9634.ckpt --denoising-weight 0.0  --model egnn --hidden-nf 128 --n-layers 10  --dataset-root /nfs/SKData/DenoisingData/qm9 > frad_egnn_h128_n10_lumo.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id frad_egnn_h128_n10_homo --dataset-arg homo --pretrained-model /mnt/nfs-ssd/data/fengshikun/FradNMI/experiments/frad_pretraining_egnn_h128_n10/step=399999-epoch=7-val_loss=0.9758-test_loss=0.9907-train_per_step=0.9634.ckpt --denoising-weight 0.0  --model egnn --hidden-nf 128 --n-layers 10  --dataset-root /nfs/SKData/DenoisingData/qm9 > frad_egnn_h128_n10_homo.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id frad_egnn_h128_n10_gap --dataset-arg gap --pretrained-model /mnt/nfs-ssd/data/fengshikun/FradNMI/experiments/frad_pretraining_egnn_h128_n10/step=399999-epoch=7-val_loss=0.9758-test_loss=0.9907-train_per_step=0.9634.ckpt --train-loss-type smooth_l1_loss  --model egnn --hidden-nf 128 --n-layers 10  --dataset-root /nfs/SKData/DenoisingData/qm9 --denoising-weight 0.0 > frad_egnn_h128_n10_gap.log 2>&1 &




# pretraining with different noisy head(use atom embedding as input)
CUDA_VISIBLE_DEVICES=3 python -u scripts/train.py --conf examples/ET-PCQM4MV2_dih_var0.04_var2_com_re.yaml --layernorm-on-vec whitened --job-id frad_pretraining_egnn_h128_n10 --num-epochs 10 --model egnn --dataset-root /mnt/nfs-ssd/data/fengshikun/DenoisingData/pcq  --hidden-nf 128 --n-layers 10 > frad_pretraining_egnn_h128_n10.log 2>&1 & 



CUDA_VISIBLE_DEVICES=4 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id egnn_frad_lumo_wonn_lr0.001 --dataset-arg lumo --pretrained-model /mnt/nfs-ssd/data/fengshikun/FradNMI/experiments/frad_pretraining_egnn/step=386103-epoch=7-val_loss=0.2119-test_loss=0.2041-train_per_step=0.1908.ckpt --denoising-weight 0.0  --model egnn --dataset-root /nfs/SKData/DenoisingData/qm9 --lr 0.001  > egnn_frad_lumo_wonn_lr0.001.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id egnn_frad_homo_wonn_lr0.001 --dataset-arg homo --pretrained-model /mnt/nfs-ssd/data/fengshikun/FradNMI/experiments/frad_pretraining_egnn/step=386103-epoch=7-val_loss=0.2119-test_loss=0.2041-train_per_step=0.1908.ckpt --denoising-weight 0.0  --model egnn --dataset-root /nfs/SKData/DenoisingData/qm9 --lr 0.001 > egnn_frad_homo_wonn_lr0.001.log 2>&1 &

CUDA_VISIBLE_DEVICES=7 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id egnn_frad_gap_wonn_lr0.001 --dataset-arg gap --pretrained-model /mnt/nfs-ssd/data/fengshikun/FradNMI/experiments/frad_pretraining_egnn/step=386103-epoch=7-val_loss=0.2119-test_loss=0.2041-train_per_step=0.1908.ckpt --train-loss-type smooth_l1_loss  --model egnn --dataset-root /nfs/SKData/DenoisingData/qm9  --denoising-weight 0.0 --lr 0.001 > egnn_frad_gap_wonn_lr0.001.log 2>&1 &

# TODO denoise weight: 0.001
CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id egnn_frad_lumo_d0.001 --dataset-arg lumo --pretrained-model /mnt/nfs-ssd/data/fengshikun/FradNMI/experiments/frad_pretraining_egnn/step=386103-epoch=7-val_loss=0.2119-test_loss=0.2041-train_per_step=0.1908.ckpt --denoising-weight 0.001  --model egnn --dataset-root /nfs/SKData/DenoisingData/qm9  > egnn_frad_lumo_d0.001.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id egnn_frad_homo_d0.001 --dataset-arg homo --pretrained-model /mnt/nfs-ssd/data/fengshikun/FradNMI/experiments/frad_pretraining_egnn/step=386103-epoch=7-val_loss=0.2119-test_loss=0.2041-train_per_step=0.1908.ckpt --denoising-weight 0.001  --model egnn --dataset-root /nfs/SKData/DenoisingData/qm9  > egnn_frad_homo_d0.001.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python -u scripts/train.py --conf examples/ET-QM9-FT_dw_0.2_long.yaml --layernorm-on-vec whitened --job-id egnn_frad_gap_d0.001 --dataset-arg gap --pretrained-model /mnt/nfs-ssd/data/fengshikun/FradNMI/experiments/frad_pretraining_egnn/step=386103-epoch=7-val_loss=0.2119-test_loss=0.2041-train_per_step=0.1908.ckpt --train-loss-type smooth_l1_loss  --model egnn --dataset-root /nfs/SKData/DenoisingData/qm9  --denoising-weight 0.001 > egnn_frad_gap_d0.001.log 2>&1 &