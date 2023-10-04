""" Here is an example of making tensorflow datasets from netCDFs. this helps with memory and stability and speed""" 

import glob 
import tensorflow as tf 
import xarray as xr 
import numpy as np 
import gc 
import tqdm 
from dask.diagnostics import ProgressBar
import pandas as pd
import copy
import sys 

print('grabbing files')
files = glob.glob('/scratch/randychase/testing_2020_cmb_*.nc')
files.sort()

# Arguments passed, this had to be chunked because datasets are HUGE 
input_array_id = int(sys.argv[1])

train_ds = xr.open_dataset(files[input_array_id],chunks={'n_samples':10})


# fill nans with minx (its so this is bascially 0 in the scaled space)
#bring into memory and fill 
train_images = train_ds.z_patch.fillna(-39.7).astype(np.float16).values 

#check max/min value
print(train_images.max(),train_images.min())

# #check for infs.
print(np.where(np.isinf(train_images)))

# # #scale inputs to 0 - 1, unstable otherwise, loss goes to infinity
#factors
maxx = 85.8
minx = -39.7

train_images = (train_images - minx) / (maxx - minx)

# #check max value
print(train_images.max(),train_images.min())

#load labels into memory
train_labels = train_ds.w_patch.astype(np.float16).values 

# #check max value because trust issues 
print(train_labels.max(),train_labels.min())

#clear up RAM 
train_ds.close()

#make tensorflow dataset 
train_dataset = tf.data.Dataset.from_tensor_slices((train_images,train_labels))

#dump to disk (this might not be experimental anymore)
tf.data.experimental.save(train_dataset, files[input_array_id][:-2]+ 'tf')

del train_dataset, train_images, train_labels,train_ds

gc.collect()