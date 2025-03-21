

time python3 train_bayesian_opt.py --dataset mnist --num_epoch 100 --batch_size 128 --lr 0.01 --gpus 1 --save_dir ./prj/exp_mnist_bayes_lenet_acc --save_model mnist_bayes_lenet_10samples_mc  --mc_samples 10 --model_name lenet --is_quant 1 --dropout_type mc --bayes_num_iter 200 --w_list "1, 0, 0, 0" 

