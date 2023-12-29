lr=$1
loss="smooth_l1_loss"
schedule=$2
dataset_index=$3
dataset_array=("AT-AT-CG-CG" "AT-AT" "Ac-Ala3-NHMe" "DHA" "buckyball-catcher" "double-walled_nanotube" "stachyose")
dataset_arg=${dataset_array[$dataset_index]}
if [ $dataset_index -eq 5 ]
then
    batch_size=16
else
    batch_size=32
fi

if [ "$schedule" == "cos" ]; then
    lr_schedule="cosine_warmup"
elif [ "$schedule" == "plat" ]; then
    lr_schedule="reduce_on_plateau"
else
    echo "Error: Invalid schedule value"
    exit 1
fi


CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py \
--conf examples/ET-MD22-angle.yaml \
--batch-size ${batch_size} \
--inference-batch-size ${batch_size} \
--num-epochs 100 \
--lr $lr \
--log-dir md22a-${dataset_arg} \
--dataset-arg ${dataset_arg} \
--ngpus 1 \
--job-id md22a_${loss}_${lr}_${lr_schedule}_${dataset_arg} \
--pretrained-model $$pretrain_model_path \
--train-loss-type ${loss}  \
--lr-schedule ${lr_schedule} \
--save-top-k 1 \
--save-interval 1 \
--test-interval 1 \
--seed 666 \
--md17 true \
--composition true \
--sep-noisy-node true \
