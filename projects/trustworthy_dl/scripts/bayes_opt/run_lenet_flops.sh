
SAVE_DIR=./prj/exp_mnist_bayes_lenet_flops
WEIGHTS="0, 0, 0, 1"
GPU_ID=1

NUM_EPOCH=100
NUM_ITER=100

echo "flops"
echo $SAVE_DIR
echo $WEIGHTS
echo $GPU_ID
echo $NUM_EPOCH
echo $NUM_ITER

time python3 train_bayesian_opt.py --dataset mnist --num_epoch $NUM_EPOCH --batch_size 128 --lr 0.01 --gpus $GPU_ID --save_dir $SAVE_DIR --save_model mnist_bayes_lenet_10samples_mc  --mc_samples 10 --model_name lenet --is_quant 0 --dropout_type mc --bayes_num_iter $NUM_ITER --w_list "$WEIGHTS" 

