
echo "dropout_rate: $1"
echo "pruning_rate: $2"


python3 train_inf_keras.py --dataset mnist --num_epoch 10 --batch_size 128 --lr 0.01 --gpus 1 --save_dir ./exp_mnist_bayes_lenet_try02 --save_model mnist_bayes_lenet_10samples_mc  --mc_samples 10 --model_name lenet --dropout_type mc  --num_bayes_layer 3 --dropout_rate $1 --p_rate $2

#echo "start to predict"
#python3 keras_pred.py --load_model ./exp_mnist_bayes_lenet_try02/mnist_bayes_lenet_10samples_mc --dataset mnist --gpus -1 --dropout_type mc --mc_samples 10 --num_bayes_layer 3 --dropout_rate $1 --p_rate $2
