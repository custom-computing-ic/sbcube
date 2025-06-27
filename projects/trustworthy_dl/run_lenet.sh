
time python3 train_bayesian_opt.py --dataset mnist --num_epoch 3 --batch_size 128 --lr 0.01 --gpus 0 --save_dir ./prj/exp_mnist_bayes_lenet_try --save_model mnist_bayes_lenet_10samples_mc  --mc_samples 10 --model_name lenet --dropout_type mc --bayes_num_iter 100  --w_list 0.25,0.25,0.25,0.25

#echo "start to predict"
#python3 keras_pred.py --load_model ./exp_mnist_bayes_lenet_try02/mnist_bayes_lenet_10samples_mc --dataset mnist --gpus -1 --dropout_type mc --mc_samples 10 --num_bayes_layer 3 --dropout_rate $1 --p_rate $2
