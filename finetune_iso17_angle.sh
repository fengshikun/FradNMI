lr=$1
loss=$2
schedule=$3


if [ "$schedule" == "cos" ]; then
    lr_schedule="cosine_warmup"
elif [ "$schedule" == "plat" ]; then
    lr_schedule="reduce_on_plateau"
else
    echo "Error: Invalid schedule value"
    exit 1
fi


CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py \
--conf examples/ET-ISO17-angle.yaml \
--batch-size 256 \
--inference-batch-size 256 \
--num-epochs 50 \
--lr $lr \
--log-dir iso-energy \
--ngpus 1 \
--job-id iso17a_${loss}_${lr}_${lr_schedule} \
--pretrained-model $pretrain_model_path \
--train-loss-type ${loss}  \
--save-top-k 1 \
--save-interval 1 \
--test-interval 1 \
--seed 666 \
--lr-schedule ${lr_schedule} \
--composition true \
--sep-noisy-node true \
--md17 true \