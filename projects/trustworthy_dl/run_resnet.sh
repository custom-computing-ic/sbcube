

time python3 train_bayesian_opt.py --dataset cifar10 --num_epoch 150 --batch_size 128 --lr 0.01 --mc_samples 10 --gpus 2 --save_model cifar_resnet_tmp_3bayeslayer_mc --model_name resnet --save_dir ./prj/exp_cifar_bayes_resnet --num_bayes_layer 3 --dropout_type mc --bayes_num_iter 50 --w_list 0.25,0.25,0.25,0.25

