
"""

The point of the script is to do a stocastic (random) hyper parameter search for the updraft work. 

This script will search the hyperparameters for the unet with *3d* convolutions and the *full dataset*. 

To allow for decent model sizes and batch sizes, this script is written to use 2 connected GPUs for synchronus training. 

"""

#GRAB 2 of the 4 gpus, meant to run on any of the 4 GPU nodes 
import py3nvml
py3nvml.grab_gpus(num_gpus=2, gpu_select=[2,3])

#imports 
import os.path
import random
import shutil
from absl import app
from absl import flags
import numpy as np
import tensorflow as tf
import xarray as xr 
from tensorboard.plugins.hparams import api as hp
import sys
import matplotlib.pyplot as plt
import io
import tensorflow_probability as tfp
import glob
import gc 

#All flag inputs that can be controlled from the script level. 
flags.DEFINE_integer(
    "num_session_groups",
    100,
    "The approximate number of session groups to create.",
)

flags.DEFINE_string(
    "logdir",
    "/tmp/hparams_demo",
    "The directory to write the summary information to.",
)
flags.DEFINE_integer(
    "summary_freq",
    600,
    "Summaries will be written every n steps, where n is the value of "
    "this flag.",
)
flags.DEFINE_integer(
    "num_epochs",
    200,
    "Number of epochs per trial.",
)

flags.DEFINE_string(
    "cache",
    '/scratch/randychase/',
    "scratch directory",
)

#data shape is 2d for 2d conv, channel dim is the max of height
INPUT_SHAPE = (128,128,1)
#parametric regression here, using SHASH
OUTPUT_CLASSES = 4 #mu, sigma, gamma, tau 

######################################################################
############################ <Hyperparams> ###########################
######################################################################
"""This is all the parameters you would like to randomly search"""

#convolution params
HP_CONV_LAYERS = hp.HParam("conv_layers", hp.Discrete([1]))
HP_CONV_KERNEL_SIZE = hp.HParam("conv_kernel_size", hp.Discrete([3, 5, 7]))
HP_CONV_ACTIVATION = hp.HParam("conv_activation", hp.Discrete(['ReLU'])) #stick to relu to enable LRP
HP_CONV_KERNELS = hp.HParam('num_of_kernels', hp.Discrete([2,4,8,16])) #these will double as you go down the unet


#unet params (depth needs to be at least 3
HP_UNET_DEPTH = hp.HParam('depth_of_unet', hp.Discrete([3,4,5]))
HP_OPTIMIZER = hp.HParam("optimizer", hp.Discrete(["adam", "rmsprop"]))
HP_LOSS = hp.HParam("loss", hp.Discrete(["RegressLogLoss_SinhArcsinh"])) 
HP_BATCHNORM = hp.HParam('batchnorm', hp.Discrete([False,True]))
HP_BATCHSIZE = hp.HParam('batch_size', hp.Discrete([64,128,256,512]))
HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([1e-3]))

#Loss params
HP_LOSS_WEIGHT = hp.HParam('weight', hp.IntInterval(1, 5)) #weight of 1 means no weights 
HP_LOSS_THRESH = hp.HParam('threshold', hp.Discrete([0,1,5,10]))

#Regularization (this helps with overfitting and XAI)
HP_L1_REG = hp.HParam('l1', hp.Discrete([1e-1,1e-2,1e-3,1e-4,0.0]))
HP_L2_REG = hp.HParam('l2', hp.Discrete([1e-1,1e-2,1e-3,1e-4,0.0]))

#fill the hparam list 
HPARAMS = [HP_CONV_LAYERS,
    HP_CONV_KERNEL_SIZE,
    HP_CONV_ACTIVATION,
    HP_CONV_KERNELS,
    HP_UNET_DEPTH,
    HP_OPTIMIZER,
    HP_LOSS,
    HP_BATCHNORM,
    HP_BATCHSIZE,
    HP_LEARNING_RATE,
    HP_LOSS_WEIGHT,
    HP_LOSS_THRESH,
    HP_L1_REG,
    HP_L2_REG,
]

######################################################################
############################ </Hyperparams> ##########################
######################################################################

######################################################################
############################# <Metrics> ##############################
######################################################################

"""This is all the metrics you want in your tensorboard"""

METRICS = [
    hp.Metric(
        "epoch_loss",
        group="validation",
        display_name="loss (val.)",
    ),
    hp.Metric(
        "epoch_loss",
        group="train",
        display_name="loss (train)",
    ),
    hp.Metric(
        "epoch_prmse",
        group="train",
        display_name="param. rmse (train)",
    ),
    hp.Metric(
        "epoch_prmse",
        group="validation",
        display_name="param. rmse (val.)",
    ),
    hp.Metric(
        "epoch_r_square",
        group="train",
        display_name="tf r_square (train)",
    ),
    hp.Metric(
        "epoch_r_square",
        group="validation",
        display_name="tf r_square (val.)",
    ),
]

######################################################################
############################# </Metrics> #############################
######################################################################


######################################################################
########################### <Functions> ##############################
######################################################################

"""These are the modular functions needed for the script/training"""

def build_loss_dict(weight,thresh):
    from custom_losses import RegressLogLoss_Normal,RegressLogLoss_SinhArcsinh
    loss_dict = {}
    #this is parametric regression that assumes normal dist.
    loss_dict['RegressLogLoss_Normal'] = RegressLogLoss_Normal(weights=[weight,1.0],thresh=thresh)
    #this is parametric regression that assumes a SHASH dist. 
    loss_dict['RegressLogLoss_SinhArcsinh'] = RegressLogLoss_SinhArcsinh(weights=[weight,1.0],thresh=thresh)
    return loss_dict

def build_opt_dict(learning_rate):
    opt_dict = {}
    opt_dict['adam'] = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    opt_dict['adagrad'] = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
    opt_dict['sgd'] = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    opt_dict['rmsprop'] = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    return opt_dict

def model_fn(hparams, seed,scope=None):
    """Create a Keras model with the given hyperparameters.
    Args:
      hparams: A dict mapping hyperparameters in `HPARAMS` to values.
      seed: A hashable object to be used as a random seed (e.g., to
        construct dropout layers in the model).
    Returns:
      A compiled Keras model.
    """

    #use keras-unet-collection to build unets 
    from keras_unet_collection import models

    rng = random.Random(seed)

    #build filter list 
    kernel_list = []
    for i in np.arange(1,hparams[HP_UNET_DEPTH]+1,1):
        kernel_list.append(hparams[HP_CONV_KERNELS]*i)

    #build unet with 3d convoltuons.
    model = models.unet_3plus_2d(INPUT_SHAPE,kernel_list,OUTPUT_CLASSES,kernel_size=hparams[HP_CONV_KERNEL_SIZE],
                      stack_num_down=hparams[HP_CONV_LAYERS], stack_num_up=hparams[HP_CONV_LAYERS],
                      activation=hparams[HP_CONV_ACTIVATION], output_activation=None, weights=None,
                      batch_norm=hparams[HP_BATCHNORM], pool='max', unpool='nearest', name='unet3d',
                      l1=hparams[HP_L1_REG], l2=hparams[HP_L1_REG])

    #get custom metrics
    from custom_metrics import ParaRootMeanSquaredError2,RSquare_Adapted

    #compile loss and optimizer 
    loss_dict = build_loss_dict(hparams[HP_LOSS_WEIGHT],hparams[HP_LOSS_THRESH])
    opt_dict = build_opt_dict(hparams[HP_LEARNING_RATE])

    #compile model with everything 
    model.compile(
        loss=loss_dict[hparams[HP_LOSS]],
        optimizer=opt_dict[hparams[HP_OPTIMIZER]],
        metrics=[ParaRootMeanSquaredError2(scope=scope),RSquare_Adapted()],
    )

    return model

def prepare_data():
    """ 
    
    This function stages the tensorflow datasets. Feasibly this can be swaped with any data loading process you want
    
    I strongly suggest tf.data.Datasets for big datasets (> 16 GB) because they are lazily loaded and can be very fast 
    
    """

    #original data is 3d, so might need to parse this in a data loader layer. 
    x_tensor_shape = (128, 128, 24, 1)
    y_tensor_shape = (128, 128, 1)
    elem_spec = (tf.TensorSpec(shape=x_tensor_shape, dtype=tf.float16), tf.TensorSpec(shape=y_tensor_shape, dtype=tf.float16))

    files = glob.glob('/ourdisk/hpc/ai2es/randychase/updraft/paper_results/data/training/training_2019_cmb_*.tf')
    files.sort()

    for i,file in enumerate(files): 
        if i == 0:
            ds_train = tf.data.Dataset.load(file,elem_spec)
        else:
            ds_train = tf.data.Dataset.concatenate(ds_train,tf.data.Dataset.load(file,elem_spec))

    print('Full training dataset size:{}'.format(ds_train.cardinality().numpy()))

    files = glob.glob('/ourdisk/hpc/ai2es/randychase/updraft/paper_results/data/validation/validation_2018_cmb_*.tf')
    files.sort()

    for i,file in enumerate(files): 
        if i == 0:
            ds_val = tf.data.Dataset.load(file,elem_spec)
        else:
            ds_val = tf.data.Dataset.concatenate(ds_val,tf.data.Dataset.load(file,elem_spec))

    print('Full validation dataset size:{}'.format(ds_val.cardinality().numpy()))
    
    return (ds_train, ds_val)

def run(data, base_logdir, session_id, hparams):
    """Run a training/validation session.
    Flags must have been parsed for this function to behave.
    Args:
      data: The data as loaded by `prepare_data()`.
      base_logdir: The top-level logdir to which to write summary data.
      session_id: A unique string ID for this session.
      hparams: A dict mapping hyperparameters in `HPARAMS` to values.
    """
    
    #define scope: 
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = model_fn(hparams=hparams, seed=session_id,scope=mirrored_strategy.scope())

    logdir = os.path.join(base_logdir, session_id)

    ds_train,ds_val = data

    #batch the training data accordingly 
    ds_train = ds_train.shuffle(ds_train.cardinality().numpy()).batch(hparams[HP_BATCHSIZE]).prefetch(tf.data.AUTOTUNE)

    #this batch is arbitrary, just needed so that you dont overwelm RAM. 
    ds_val = ds_val.batch(512).prefetch(tf.data.AUTOTUNE)

    #need to map the data into dropping the 1 dim 
    colmax = tf.keras.Sequential([tf.keras.layers.MaxPool3D(pool_size=(1, 1, 24)),
                             tf.keras.layers.Reshape((128,128,1),input_shape=(128, 128, 1, 1))])
    ds_train = ds_train.map(lambda x_img,y_img: (colmax(x_img), y_img), num_parallel_calls= tf.data.AUTOTUNE)
    ds_val = ds_val.map(lambda x_img,y_img: (colmax(x_img), y_img), num_parallel_calls= tf.data.AUTOTUNE)

    callback = tf.keras.callbacks.TensorBoard(
        logdir,
        update_freq='epoch',
        profile_batch=0,  # workaround for issue #2084
    )

    hparams_callback = hp.KerasCallback(logdir, hparams)

    callback_es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7,restore_best_weights=True)
    
    #should kill it if there is an especially bad training loss... (i.e., nan; no need to waste time) 
    nan_callback = tf.keras.callbacks.TerminateOnNaN()

    #add images to board 
    print(model.summary())
    result = model.fit(ds_train,
        epochs=flags.FLAGS.num_epochs,
        shuffle=False,
        validation_data=ds_val,
        callbacks=[callback, hparams_callback,callback_es,nan_callback],verbose=0)

    #save trained model, need to build path first 
    split_dir = logdir.split('paper_results')
    split_dir2 = split_dir[1].split('logs_2d_max')
    right = split_dir2[0][:-1] + split_dir2[1]
    left = '/ourdisk/hpc/ai2es/randychase/updraft/paper_results/models_2d_max/'
    model.save(left + right + "_model.h5")

    #do some cleanup
    del model,ds_train,ds_val

    gc.collect()




def run_all(logdir, verbose=False):
    """Perform random search over the hyperparameter space.
    Arguments:
      logdir: The top-level directory into which to write data. This
        directory should be empty or nonexistent.
      verbose: If true, print out each run's name as it begins.
    """
    data = prepare_data()
    rng = random.Random(42) #changed this seed to get a new param set  

    with tf.summary.create_file_writer(logdir).as_default():
        hp.hparams_config(hparams=HPARAMS, metrics=METRICS)

    sessions_per_group = 1
    num_sessions = flags.FLAGS.num_session_groups * sessions_per_group
    session_index = 0  # across all session groups
    for group_index in range(flags.FLAGS.num_session_groups):
        hparams = {h: h.domain.sample_uniform(rng) for h in HPARAMS}
        hparams_string = str(hparams)
        for repeat_index in range(sessions_per_group):
            session_id = str(session_index)
            session_index += 1
            if session_index <= 85:
                continue
            else:
                if verbose:
                    print(
                        "--- Running training session %d/%d"
                        % (session_index, num_sessions)
                    )
                    print(hparams_string)
                    print("--- repeat #: %d" % (repeat_index + 1))
                run(
                    data=data,
                    base_logdir=logdir,
                    session_id=session_id,
                    hparams=hparams,
                )


def main(unused_argv):
    np.random.seed(42)
    logdir = flags.FLAGS.logdir
    # print('removing old logs')
    # shutil.rmtree(logdir, ignore_errors=True)
    print("Saving output to %s." % logdir)
    run_all(logdir=logdir, verbose=True)
    print("Done. Output saved to %s." % logdir)


if __name__ == "__main__":
    app.run(main)