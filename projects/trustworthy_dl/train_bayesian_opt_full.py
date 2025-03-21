#!/usr/bin/env python3
import sys
sys.path.append(sys.path[0] + '/converter/keras')
sys.path.append(sys.path[0] + '/models')

from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from tensorflow.keras.layers import *
from qkeras.qlayers import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu
from tensorflow.keras.models import load_model
from qkeras.utils import _add_supported_quantized_objects
from tensorflow.keras.utils import to_categorical
from qkeras import *
from tensorflow.keras.optimizers import Adam, SGD
import os
import argparse
import numpy as np
import random
import time
from models import lenet, ResNet18, VGG11
from converter.keras.MCDropout import MCDropout, BayesianDropout
from converter.keras.Masksembles import MasksemblesModel, Masksembles
from converter.keras.train import mnist_data
from qmodels import Qlenet, QResNet18, QVGG11, QVIBNN
import keras
import tensorflow_probability as tfp
import tensorflow as tf
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from data_utils import CIFAR10Data
from data_utils import random_noise_data
from scipy.io import loadmat
from svhn.utils import CosineAnnealingScheduler
#from model_utils import Top_Level_Model
from metric_utils import *
from model_utils import *

import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from tensorflow_model_optimization.sparsity.keras import strip_pruning
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from tensorflow.keras.callbacks import ModelCheckpoint
from bayes_opt import BayesianOptimization as BOptimize
import pandas as pd

from keras_flops import get_flops

model_num_layer = {"lenet": 3, "resnet": 3}

def list_of_floats(arg):
    return list(map(float, arg.split(',')))


#if __name__ == '__main__':
# Let's allow the user to pass the filename as an argument
parser = argparse.ArgumentParser()

parser.add_argument("--dataset", default="mnist", type=str, required=True, help="Name of dataset")
parser.add_argument("--save_dir", default="./prj/exp_mnist_bayes_lenet", type=str, required=True, help="Directory name of saved results")
parser.add_argument("--model_name", default="lenet", type=str, help="Name of contructed model")

parser.add_argument("--gpus", default="0,1", type=str, required=True, help="GPUs id, separated by comma without space, e.g, 0,1,2")
parser.add_argument("--save_model", default=None, type=str, help="Name of save model")
parser.add_argument("--is_train", default=1, type=int, help="Whether to train model")
parser.add_argument("--is_quant", default=0, type=int, help="Whether to quantize model")
parser.add_argument("--load_model", default=None, type=str, help="Name of load model")

parser.add_argument("--validation_split", default=0.1, type=float, help="Validation slipt of dataset")
parser.add_argument("--lr", default=0.01, type=float, help="Learning rate")
parser.add_argument("--num_epoch", default=100, type=int, required=True, help="The number of epoch for training")
parser.add_argument("--num_bayes_layer", default=0, type=int, help="The number of Bayesian Layer")
parser.add_argument("--quant_tbit", default=6, type=int, help="The total bits of quant")
parser.add_argument("--quant_ibit", default=0, type=int, help="The integer bits of quant")
parser.add_argument("--dropout_rate", default=0.2, type=float, help="The dropout rate")
parser.add_argument("--num_masks", default=4, type=int, help="The number of masks")
parser.add_argument("--scale", default=4, type=float, help="The scale")
parser.add_argument("--mc_samples", default=5, type=int, help="The number of MC samples")

parser.add_argument("--batch_size", default=64, type=int, help="The number of batches for training")
parser.add_argument("--dropout_type", default="mc", type=str, choices=["mc", "mask"], help="Dropout type, Monte-Carlo Dropout (mc) or Mask Ensumble (mask)")
parser.add_argument("--is_me", default=0, type=int, help="Whether use multi-exit, 0 denote no use")
parser.add_argument("--num_exits", default=2, type=int, help="The number of exits in multi-exit arch")
parser.add_argument("--p_rate", default=0.0, type=float, help="pruning rate")
parser.add_argument("--seed", default=0, type=int, help="seed")
parser.add_argument("--num_eval_images", default=200, type=int, help="The number of evaluated images")
parser.add_argument("--num_bins", default=10, type=int, help="The number of bins while calculating ECE")
parser.add_argument("--bayes_num_iter", default=2, type=int, help="The number of bins while calculating ECE")
parser.add_argument("--scale_factor", default=1.0, type=float, help="")
parser.add_argument("--w_list", type=list_of_floats, help="")

args = parser.parse_args()

# random seed
random.seed(args.seed)
np.random.seed(args.seed)
keras.utils.set_random_seed(args.seed)
tf.random.set_seed(args.seed)

# Set GPU environment
gpus = args.gpus.split(",")
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpus)

start_time = time.strftime("%Y%m%d-%H%M%S")




#if not os.path.exists(args.save_dir):
#    print ("Create Non-exiting Directory")
args.save_dir = args.save_dir + start_time
os.makedirs(args.save_dir)


def get_dataset(args):
    if args.dataset == "mnist":
        num_classes = 10
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        RESHAPED = 784

        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")

        x_train = x_train[..., np.newaxis]
        x_test = x_test[..., np.newaxis]

        x_train /= 256.0
        x_test /= 256.0
        y_train = to_categorical(y_train, num_classes)
        y_test = to_categorical(y_test, num_classes)
    elif args.dataset == "cifar10":
        cifar10_data = CIFAR10Data()
        x_train, y_train, x_test, y_test = cifar10_data.get_data(subtract_mean=True)

        num_train = int(x_train.shape[0] * 0.9)
        num_val = x_train.shape[0] - num_train
        mask = list(range(num_train, num_train+num_val))
        x_val = x_train[mask]
        y_val = y_train[mask]

        mask = list(range(num_train))
        x_train = x_train[mask]
        y_train = y_train[mask]

        print('num train:%d num val:%d' % (num_train, num_val))
        data = (x_train, y_train, x_val, y_val, x_test, y_test)
    elif args.dataset == "svhn":
        # Pre-processing, get from https://www.kaggle.com/code/dimitriosroussis/svhn-classification-with-cnn-keras-96-acc
        np.random.seed(20)
        train_raw = loadmat('./svhn/train_32x32.mat')
        test_raw = loadmat('./svhn/test_32x32.mat')
        train_images = np.array(train_raw['X'])
        test_images = np.array(test_raw['X'])

        train_labels = train_raw['y']
        test_labels = test_raw['y']
        train_images = np.moveaxis(train_images, -1, 0)
        test_images = np.moveaxis(test_images, -1, 0)
        train_images = train_images.astype('float32')
        test_images = test_images.astype('float32')
        train_labels = train_labels.astype('int32')
        test_labels = test_labels.astype('int32')
        train_images /= 255.0
        test_images /= 255.0
        from sklearn.preprocessing import LabelBinarizer
        lb = LabelBinarizer()
        train_labels = lb.fit_transform(train_labels)
        test_labels = lb.fit_transform(test_labels)
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(train_images, train_labels,
                                                        test_size=0.15, random_state=22)
    else:
        raise NotImplementedError("Dataset not supoorted")

    if args.dataset == "cifar10":
        dataset = {"x_train": x_train, "x_test": x_test, "y_train": y_train, "y_test": y_test, "x_val": x_val, "y_val": y_val}
    else:
        dataset = {"x_train": x_train, "x_test": x_test, "y_train": y_train, "y_test": y_test}
    return dataset

def get_model(args):
    #model=Sequential()
    if args.model_name == "lenet":
        if args.is_quant != 0:
            model = Qlenet(args, model_num_layer[args.model_name])
        else:
            model = lenet(args, model_num_layer[args.model_name])
    elif args.model_name == "vibnn": # vibnn is deprecated as we found its permance is very bad in HLS4ML
        model = QVIBNN(args)
    elif args.model_name == "resnet":
        if args.is_quant != 0:
            model = QResNet18(input_shape=(32, 32, 3), classes=10, args=args, weight_decay=1e-4, base_filters=64)
        else:
            model = ResNet18(input_shape=(32, 32, 3), classes=10, args=args, weight_decay=1e-4, base_filters=64)
    elif args.model_name == "vgg":
        if args.is_quant != 0:
            model = QVGG11(args, filters=16, dense_out=[16, 16, 10])
        else:
            model = VGG11(args, filters=16, dense_out=[16, 16, 10])
    else:
        raise NotImplementedError("Model not supoorted")
    #model.compile(optimizer=SGD(lr = args.lr), loss=['categorical_crossentropy'], metrics=['accuracy'])
    print(model.summary())
    return model


def train(args, model, dataset):

    if args.model_name == "lenet" or args.model_name == "vibnn": # vibnn is deprecated as we found its permance is very bad in HLS4ML
        #chkp = ModelCheckpoint(
        #    args.save_dir
        #    + "/"
        #    + "best_chkp.tf",
        #    monitor="val_loss",
        #    verbose=1,
        #    save_best_only=True,
        #    save_weights_only=False,
        #    mode="auto",
        #    save_freq="epoch",
        #)

        if args.p_rate != 0.0:
            #callbacks = [chkp, pruning_callbacks.UpdatePruningStep() ]
            callbacks = [pruning_callbacks.UpdatePruningStep() ]
        else:
            #callbacks = [chkp]
            callbacks = []

        train_stat = model.fit(
            dataset['x_train'], dataset['y_train'], batch_size=args.batch_size,
            epochs=args.num_epoch, initial_epoch=1,
            validation_split=args.validation_split, callbacks=callbacks)
#        train_stat = model.fit(
#            dataset['x_train'], dataset['y_train'], batch_size=args.batch_size,
#            epochs=args.num_epoch, initial_epoch=1,
#            validation_split=args.validation_split)

    elif args.model_name == "resnet":
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            # rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=4,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=4,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False,  # randomly flip images
        )
        print('train with data augmentation')
        train_gen = datagen.flow(dataset['x_train'], dataset['y_train'], batch_size=args.batch_size)
        # def lr_scheduler(epoch):
        #     lr = args.lr
        #     new_lr = lr
        #     if epoch <= 91:
        #         pass
        #     elif epoch > 91 and epoch <= 137:
        #         new_lr = lr * 0.1
        #     else:
        #         new_lr = lr * 0.01
        #     print('new lr:%.2e' % new_lr)
        #     return new_lr
        def lr_scheduler(epoch):
            lr = args.lr
            new_lr = lr * (0.1 ** (epoch // 50))
            print('new lr:%.2e' % new_lr)
            return new_lr
        reduce_lr = CosineAnnealingScheduler(T_max=args.num_epoch, eta_max=args.lr, eta_min=1e-4)

        if args.p_rate != 0.0:
            callbacks = [reduce_lr, pruning_callbacks.UpdatePruningStep() ]
        else:
            callbacks = [reduce_lr]
        history = model.fit_generator(generator=train_gen,
                                           epochs=args.num_epoch,
                                           callbacks=callbacks,
                                           validation_data=(dataset['x_val'], dataset['y_val']),
                                           )

    elif args.model_name == "vgg":
        callbacks = [CosineAnnealingScheduler(T_max=args.num_epoch, eta_max=args.lr, eta_min=1e-4)]
        datagen = ImageDataGenerator(rotation_range=8,
                                    zoom_range=[0.95, 1.05],
                                    height_shift_range=0.10,
                                    shear_range=0.15)
        train_stat = model.fit(datagen.flow(dataset['x_train'], dataset['y_train'], batch_size=args.batch_size),
            epochs=args.num_epoch, validation_data=(dataset['x_test'], dataset['y_test']), callbacks=callbacks)
    else:
        raise NotImplementedError("Training not supoorted")

def keras_pred(args, model, dataset):
    #co = {"BayesianDropout": BayesianDropout, "MCDropout": MCDropout, "Masksembles": Masksembles, "MasksemblesModel": MasksemblesModel}
    co = {  "BayesianDropout": BayesianDropout,
            "MCDropout": MCDropout,
            "Masksembles": Masksembles,
            "MasksemblesModel": MasksemblesModel,
            "PruneLowMagnitude": pruning_wrapper.PruneLowMagnitude
            }
    _add_supported_quantized_objects(co)
    #model = load_model(args.load_model + '.h5', custom_objects=co)
    #model = Top_Level_Model(args, model)

    model.model.summary()
    check_sparsity(model.model)
    print("dropout_rate", args.dropout_rate)
    model.model  = strip_pruning(model.model)

    from train_qkeras_mcme import get_dataset
    data_dict = get_dataset(args)

    #x_test_random = random_noise_data("mnist")[:args.num_eval_images]
    x_test_random_full = random_noise_data(args.dataset)

    from sklearn.metrics import accuracy_score

    y_prob        = model.predict(data_dict["x_test"])

    y_logits    = np.log(y_prob/(1-y_prob + 1e-15))
    ece_keras   = tfp.stats.expected_calibration_error(num_bins=args.num_bins,
        logits=y_logits, labels_true=np.argmax(data_dict["y_test"],axis=1), labels_predicted=np.argmax(y_prob,axis=1))

    accuracy_keras = float(accuracy_score (np.argmax(data_dict["y_test"],axis=1), np.argmax(y_prob,axis=1)))
    entropy_keras = entropy(model.predict(np.ascontiguousarray(x_test_random_full)))
    #print("Full dataset, Accuracy Keras:  {}, ECE Keras {}, aPE Keras {}".format(accuracy_keras, ece_keras, entropy_keras))

    return accuracy_keras, float(ece_keras), entropy_keras



def Prune_Model(args, model, x_train_len):

    NSTEPS =   int(x_train_len)  // args.batch_size

    def pruneFunction(layer):
        pruning_params = {
            'pruning_schedule': sparsity.PolynomialDecay(
                initial_sparsity=0.0, final_sparsity=args.p_rate, begin_step=NSTEPS * 2, end_step=NSTEPS * 8, frequency=NSTEPS
            )
        }
        if isinstance(layer, tf.keras.layers.Conv2D):
            print("prune Conv2D")
            return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)

        if isinstance(layer, tf.keras.layers.Dense) and layer.name != 'fc_2': # exclude output_dense
            return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
        return layer

    #print_qmodel_summary(model)
    model = tf.keras.models.clone_model(model, clone_function=pruneFunction)

    model.compile(optimizer=SGD(lr = args.lr), loss=['categorical_crossentropy'], metrics=['accuracy'])
    return model

    #pr = pruning_callbacks.UpdatePruningStep()

# Get dataset and model


discrete_options = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.80, 0.85, 0.90, 0.95]
discrete_layer = [1,2,3]

bayesian_output_data = {
    "num_iter": [],
    "dropout_rate": [],
    "pruning_rate": [],
    "num_bayes_layer": [],
    "scale_factor": [],
    "mc_samples": [],
    "accuracy": [],
    "flops": [],
    "ECE": [],
    "aPE": [],
    "score": []
}
bayesian_output_data = pd.DataFrame(bayesian_output_data)

num_iter = 0

acc_base = {
"lenet": 0.99,
"resnet": 0.92
}

ape_base ={
"lenet": 1.5,
"resnet": 1.8
}

ece_base ={
"lenet": 0.05,
"resnet": 0.09
}

flops_base = {
"lenet": 5340192,
"resnet": 1112422588
}

#w_base = {
#"acc"  : 0.25 ,
#"ape"  : 0.25 ,
#"ece"  : 0.25 ,
#"flops": 0.25 ,
#}

w_base = {
"acc"  : args.w_list[0] ,
"ape"  : args.w_list[1] ,
"ece"  : args.w_list[2] ,
"flops": args.w_list[3] ,
}


def black_box_function(dropout_rate, p_rate, num_bayes_layer, scale_factor, mc_samples ):
    def params_discrete(param):
        param = int(param)
        return discrete_options[param]
    def params_discrete_layer(param):
        param = int(param)
        return discrete_layer[param]

    global args
    global num_iter
    args.dropout_rate = params_discrete(dropout_rate)
    args.p_rate       = params_discrete(p_rate)
    args.scale_factor = params_discrete(scale_factor)
    #args.dropout_rate = float(dropout_rate)
    #args.p_rate       = float(p_rate)
    args.num_bayes_layer = params_discrete_layer(num_bayes_layer)
    args.mc_samples   = int(mc_samples)

    print("dropout_rate", args.dropout_rate )
    print("pruning_rate", args.p_rate )
    print("num_bayes_layer", args.num_bayes_layer )
    print("scale_factor", args.scale_factor )
    print("mc_samples", args.mc_samples )

    save_model = args.save_dir + "/" + args.save_model + "_" + f"iter{num_iter}"
    dataset = get_dataset(args)
    model = get_model(args)

    flops = get_flops(model, batch_size=1)
    print(f"FLOPS: {flops / 10 ** 6:.03} M")

    if(args.p_rate!=0.0):
        model = Prune_Model(args, model, len(dataset["x_train"]))
    model = Top_Level_Model(args, model)

    flops = int(flops * (1 - args.p_rate)) * args.mc_samples
    print(f"FLOPS after prune: {flops / 10 ** 6:.03} M")
    #exit(0)

    train(args, model, dataset)

    model.model.summary()
    scores = model.evaluate(dataset["x_test"], dataset["y_test"], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    if save_model is not None:
        model.model.save(save_model+'.h5')

    accuracy, ece, ape = keras_pred(args, model, dataset)
    print("Full dataset, Accuracy Keras:  {}, ECE Keras {}, aPE Keras {}".format(accuracy, ece, ape))

    final_score = float(accuracy / acc_base[args.model_name])   * w_base["acc"] \
                + float(ape      / ape_base[args.model_name])   * w_base["ape"] \
                - float(ece      / ece_base[args.model_name])   * w_base["ece"] \
                - float(flops    / flops_base[args.model_name]) * w_base["flops"]

    #print("acc_score",float(accuracy/acc_base[args.model_name]) * w_base["acc"] )
    #print("ape_score",float(ape/ape_base[args.model_name]) * w_base["ape"] )
    #print("flops_score",float(flops/flops_base[args.model_name]) * w_base["flops"])
    #print("total_score", final_score)
    #exit(0)
    if (accuracy < 0.95) & (args.model_name == "lenet") :
        final_score = -sys.maxsize
    elif (accuracy < 0.85) & (args.model_name == "resnet") :
        final_score = -sys.maxsize
    #elif (ece > 0.05) & (args.model_name == "lenet") :
    #    final_score = -sys.maxsize
    #elif (ece > 0.10) & (args.model_name == "resnet") :
    #    final_score = -sys.maxsize
    #else:
    #    #final_score = ape
    #    #final_score = -ece
    #    final_score = -flops


    global bayesian_output_data
    bayesian_output_data.loc[len(bayesian_output_data.index)] = [
        int(num_iter),
        args.dropout_rate,
        args.p_rate,
        args.num_bayes_layer,
        args.scale_factor,
        args.mc_samples,
        accuracy,
        flops,
        ece,
        ape,
        final_score]

    num_iter += 1
    return final_score


params_nn ={
    'dropout_rate': (0, len(discrete_options)-0.001),
    'p_rate'      : (0, len(discrete_options)-0.001) ,
    #'dropout_rate': (0.1, 0.7),
    #'p_rate'      : (0.1, 0.95) ,
    'num_bayes_layer': (0, len(discrete_layer)-0.001),
    'scale_factor'    : (0, len(discrete_options)-0.001) ,
    'mc_samples'  : (3, 100-0.001)
}

optimizer = BOptimize(
    f=black_box_function,
    pbounds=params_nn,
    random_state=1,
    allow_duplicate_points=True
)

optimizer.maximize(init_points=1, n_iter=args.bayes_num_iter)

save_csv_loc = args.save_dir + "/bayesian_opt" + "_" + start_time + ".csv"
bayesian_output_data.to_csv(save_csv_loc, index=False)
print("\n")
print(bayesian_output_data)


params_nn_ = optimizer.max['params']
print("\n")
print("best params:")
#print("dropout_rate:", params_nn_['dropout_rate'])
#print("Pruning_rate:", params_nn_['p_rate'])
print("dropout_rate:", discrete_options[int(params_nn_['dropout_rate'])])
print("Pruning_rate:", discrete_options[int(params_nn_['p_rate'])])
print("Num_bayes_layer:", discrete_layer[int(params_nn_['num_bayes_layer'])] )
print("scale_factor:",  discrete_options[int(params_nn_['scale_factor'])])
print("mc_samples:",  int(params_nn_['mc_samples']))




