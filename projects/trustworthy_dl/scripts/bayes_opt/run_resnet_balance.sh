


SAVE_DIR=./prj/exp_cifar_bayes_resnet_balance
WEIGHTS="0.25, 0.25, 0.25, 0.25"
GPU_ID=2

NUM_EPOCH=200
NUM_ITER=100

echo "balance"
echo $SAVE_DIR
echo $WEIGHTS
echo $GPU_ID
echo $NUM_EPOCH
echo $NUM_ITER

time python3 train_bayesian_opt.py --dataset cifar10 --num_epoch $NUM_EPOCH  --batch_size 128 --lr 0.01 --mc_samples 10 --gpus $GPU_ID --save_model cifar_resnet_tmp_3bayeslayer_mc --model_name resnet --save_dir $SAVE_DIR  --num_bayes_layer 3 --dropout_type mc --bayes_num_iter $NUM_ITER --w_list "$WEIGHTS"

