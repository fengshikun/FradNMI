lr=$1
loss=$2
device=$3
schedule=$4


if [ "$schedule" == "cos" ]; then
    lr_schedule="cosine_warmup"
elif [ "$schedule" == "plat" ]; then
    lr_schedule="reduce_on_plateau"
else
    echo "Error: Invalid schedule value"
    exit 1
fi


CUDA_VISIBLE_DEVICES=$device python -u scripts/train.py \
--conf examples/ET-ANI1X-angle.yaml \
--batch-size 32 \
--inference-batch-size 64 \
--num-epochs 10 \
--lr $lr \
--log-dir ani1x-energy \
--ngpus 1 \
--job-id ani1xa_${loss}_${lr}_${lr_schedule} \
--pretrained-model experiments/step=399999-epoch=8-val_loss=0.1626-test_loss=0.2945-train_per_step=0.1681.ckpt \
--train-loss-type $loss  \
--save-top-k 1 \
--save-interval 1 \
--test-interval 1 \
--composition true \
--sep-noisy-node true \
--lr-schedule ${lr_schedule} \
--seed 666 \
--md17 true \