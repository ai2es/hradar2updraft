""" 

This script was used to get the test stats reported in the paper. Note, filepaths from my original machine have been removed

Note, this code was never run as a whole. Parts were commented out and re-run to do different things. 
This was done because of scripting that was required on the OU machine and i didnt want to make a new driver script and python script.

The test data are hosted on zenodo because of size... Train and Val sets are not available online because they are HUGE. 

 """ 

 ################ BIT ONE ########################
#imports 
import numpy as np
import tensorflow as tf
import xarray as xr 
import tensorflow_probability as tfp
import glob
import tqdm
import gc 

# grab GPU(s) [runs faster with 2]
import py3nvml
py3nvml.grab_gpus(num_gpus=2, gpu_select=[0,1])

# load model (need to build this so it takes it from the top level; i.e. input to script)

# 3d 1 feature model 
model = tf.keras.models.load_model('//PATH//TO//MODELS//model_3d.h5',compile=False)

# 2d 24 feature model 
model = tf.keras.models.load_model('//PATH//TO//MODELS//model_2d24f.h5',compile=False)

# 2d 1 feature model 
model = tf.keras.models.load_model(''//PATH//TO//MODELS//model_2dmax.h5'',compile=False)

# outer loop, over val files
x_tensor_shape = (128, 128, 24, 1)
y_tensor_shape = (128, 128, 1)
elem_spec = (tf.TensorSpec(shape=x_tensor_shape, dtype=tf.float16), tf.TensorSpec(shape=y_tensor_shape, dtype=tf.float16))

files = glob.glob('//PATH//TO//DATA//testing_2020_cmb_*.tf')
files.sort()
for i,file in enumerate(files): 
    ds_val = tf.data.Dataset.load(file,elem_spec)

    y_np = np.zeros([ds_val.cardinality().numpy(),128,128,1],dtype=np.float16)
    for ii,(x,y) in enumerate(tqdm.tqdm(ds_val)):
        y_np[ii] = y

    #set arbitrary batch 
    ds_val = ds_val.batch(64)

    #do nothing if 3d model is chosen ;; 
    
    #2d 24 feature model ;; need to reshape to add last dim for proper running with channels last
    # reshape_x = tf.keras.Sequential([tf.keras.layers.Reshape((128,128,24),input_shape=(128, 128, 24, 1))])
    # ds_val = ds_val.map(lambda x_img,y_img: (reshape_x(x_img), y_img), num_parallel_calls= tf.data.AUTOTUNE)

    #2d 1 feature model ;; need to map the data into dropping the 1 dim 
    colmax = tf.keras.Sequential([tf.keras.layers.MaxPool3D(pool_size=(1, 1, 24)),
                             tf.keras.layers.Reshape((128,128,1),input_shape=(128, 128, 1, 1))])
    ds_val = ds_val.map(lambda x_img,y_img: (colmax(x_img), y_img), num_parallel_calls= tf.data.AUTOTUNE)

    #get predictions 
    y_hat = model.predict(ds_val)

    print('change dtype')
    #unpack preds 
    y_pred = tf.cast(y_hat, tf.float64)
    
    #Chase adaptation to prevent bad inital params 
    root_power = tf.constant(1.,tf.float64)/tf.math.multiply(tf.constant(10.,tf.float64),tf.cast(tf.math.exp(1.),tf.float64))

    #these are the 4 parameters of the dist.
    mu = y_pred[...,0] 
    sigma = tf.math.pow(tf.math.exp(y_pred[...,1]),root_power)
    gamma = y_pred[..., 2] 
    tau = tf.math.pow(tf.math.exp(y_pred[...,3]),root_power)

    #build dists.
    cond_dist = tfp.distributions.SinhArcsinh(mu, sigma,skewness=gamma,tailweight=tau)

    #get median pred
    y_out = cond_dist.quantile(0.5)

    #get pit distribution 
    pit = cond_dist.cdf(y_np[...,0])

    #get IQR
    Q1 = cond_dist.quantile(0.25)
    Q2 = cond_dist.quantile(0.75)

    #dump data to file 
    da_med = xr.DataArray(y_out.numpy().astype(np.float32),dims=['n_samples','x','y'],name='median')
    da_Q1 = xr.DataArray(Q1.numpy().astype(np.float32),dims=['n_samples','x','y'],name='Q1')
    da_Q2 = xr.DataArray(Q2.numpy().astype(np.float32),dims=['n_samples','x','y'],name='Q2')
    da_pit = xr.DataArray(pit.numpy().astype(np.float32),dims=['n_samples','x','y'],name='pit')
    da_true = xr.DataArray(y_np[...,0].astype(np.float32),dims=['n_samples','x','y'],name='y_true')

    ds = xr.merge([da_med,da_Q1,da_Q2,da_pit,da_true])

    ds.to_netcdf('/PATH/TESTS_{}_3d.nc'.format(i))

    #clean up 
    del y_out,y_hat,cond_dist,ds_val,y_pred,mu,sigma,gamma,tau,Q1,Q2,pit

    gc.collect()
################ /BIT ONE ########################

################ BIT TWO ########################
# get probabalistic metrics and pit diagrams
#################################################
#once the above part is run, go ahead and run this 
# files = glob.glob('//PATH//TOO//ABOVE//OUTPUT//TESTS_*.nc')
# files.sort()

# ds = xr.open_mfdataset(files,concat_dim='n_samples',combine='nested')
# print(ds)

# print('max value:{}'.format(ds.y_true.max().values))

# IoU = tf.keras.metrics.IoU(num_classes=2,target_class_ids=[1])
# y_true = ds.y_true.values
# y_pred = ds['median'].values
# threshs = np.arange(1,11,1,dtype='int')
# for ii,thresh in enumerate(threshs):
#     IoU.reset_state()

#     var_tmp = np.zeros(y_true.shape,dtype='int')
#     var_tmp[y_true >= thresh] = 1

#     var_tmp2 = np.zeros(y_pred.shape,dtype='int')
#     var_tmp2[y_pred >= thresh] = 1
#     IoU.update_state(var_tmp,var_tmp2)
#     print(thresh,IoU.result())

# # print(IoU.result())

# del y_true, y_pred, IoU, var_tmp,var_tmp2

# y_np = ds.y_true.values
# Q1 = ds.Q1.values 
# Q2 = ds.Q2.values 
# print('running wheres')

# #calc frequency the 'truth' shows up in the IQR
# left = np.where(y_np.ravel() > Q1.ravel())
# right = np.where(y_np.ravel() < Q2.ravel())
# both = np.intersect1d(left,right)

# #print to out
# print('truth in IQR: {}'.format(len(both)/len(y_np.ravel())))

# del y_np,Q1,Q2

# pit = ds.pit.values

# plot PIT diagram
# import matplotlib.colors as colors
# import matplotlib.cm as cmx
# import matplotlib 
# import matplotlib.patheffects as path_effects
# import cmocean
# import matplotlib.pyplot as plt

# import matplotlib

# #plot parameters that I personally like, feel free to make these your own.
# matplotlib.rcParams['axes.facecolor'] = [0.9,0.9,0.9]
# matplotlib.rcParams['axes.labelsize'] = 14
# matplotlib.rcParams['axes.titlesize'] = 14
# matplotlib.rcParams['xtick.labelsize'] = 12
# matplotlib.rcParams['ytick.labelsize'] = 12
# matplotlib.rcParams['legend.fontsize'] = 12
# matplotlib.rcParams['legend.facecolor'] = 'w'
# matplotlib.rcParams['savefig.transparent'] = False


# cmap = cmocean.cm.balance
# #normalize colorscale to vmin and vmax (for ref try 0 and 70)
# cNorm  = colors.Normalize(vmin=0, vmax=1)
# scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

# C,b = np.histogram(pit.ravel(),np.arange(0,1.1,0.1))
# midpoints = np.arange(0.1,1.1,0.1)-0.05
# plt.figure(facecolor='w',)
# for ii in np.arange(len(midpoints)):
#     plt.bar(midpoints[ii],C[ii]/C.sum(),width=0.1,edgecolor='k',color=scalarMap.to_rgba(midpoints[ii]))
# plt.axhline(0.1,ls='--',color='k')
# plt.xlabel('PIT')
# plt.ylabel('probability')
# plt.ylim([0,0.16])
# plt.tight_layout()

# #need to make this savestring match the model name 
# plt.savefig('/PATH/PIT_test_2dmax.png',dpi=300)

# #Get PIT D statistic 
# D = 0
# for i in np.arange(0,10):
#     D += ((C/C.sum())[i] - 1/10)**2
    
# D = D/10
# D = np.sqrt(D)
# print('D: {}'.format(D))

# ds.close()

# del pit,ds

################ /BIT TWO ########################

################ BIT THREE ########################
# deterministic outputs, one-to-one plots 
###################################################

# get scatter plot 
# def boxbin(x,y,xedge,yedge,c=None,figsize=(5,5),cmap='viridis',mincnt=10,vmin=None,vmax=None,edgecolor=None,powernorm=False,
#            ax=None,normed=False,method='mean',quantile=None,alpha=1.0,cbar=True,unconditional=False,master_count=np.array([])):
    
#     """ This function will grid data for you and provide the counts if no variable c is given, or the median if 
#     a variable c is given. In the future I will add functionallity to do the median, and possibly quantiles. 
    
#     x: 1-D array 
#     y: 1-D array 
#     xedge: 1-D array for xbins 
#     yedge: 1-D array for ybins
    
#     c: 1-D array, same len as x and y 
    
#     returns
    
#     axis handle 
#     cbar handle 
#     C matrix (counts or median values in bin)
    
#     """
    
#     midpoints = np.empty(xedge.shape[0]-1)
#     for i in np.arange(1,xedge.shape[0]):
#         midpoints[i-1] = xedge[i-1] + (np.abs(xedge[i] - xedge[i-1]))/2.
    
#     #note on digitize. bin 0 is outside to the left of the bins, bin -1 is outside to the right
#     ind1 = np.digitize(x,bins = xedge) #inds of x in each bin
#     ind2 = np.digitize(y,bins = yedge) #inds of y in each bin
    
    
#     #drop points outside range 
#     outsideleft = np.where(ind1 != 0)
#     ind1 = ind1[outsideleft]
#     ind2 = ind2[outsideleft]
#     if c is None:
#         pass
#     else:
#         c = c[outsideleft]
        
#     outsideright = np.where(ind1 != len(xedge))
#     ind1 = ind1[outsideright]
#     ind2 = ind2[outsideright]
#     if c is None:
#         pass
#     else:
#         c = c[outsideright]
        
#     outsideleft = np.where(ind2 != 0)
#     ind1 = ind1[outsideleft]
#     ind2 = ind2[outsideleft]
#     if c is None:
#         pass
#     else:
#         c = c[outsideleft]
#     outsideright = np.where(ind2 != len(yedge))
#     ind1 = ind1[outsideright]
#     ind2 = ind2[outsideright]
#     if c is None:
#         pass
#     else:
#         c = c[outsideright]
    

#     if c is None:
#         c = np.zeros(len(ind1))
#         df = pd.DataFrame({'x':ind1-1,'y':ind2-1,'c':c})
#         df2 = df.groupby(["x","y"]).count()
#         df = df2.where(df2.values >= mincnt).dropna()
#         C = np.ones([xedge.shape[0]-1,yedge.shape[0]-1])*-9999
#         for i,ii in enumerate(df.index.values):
#             C[ii[0],ii[1]] = df.c.values[i]
#         C = np.ma.masked_where(C == -9999,C)
        
#         if normed:
#             n_samples = np.ma.sum(C)
#             C = C/n_samples
#             C = C*100
#             print('n_samples= {}'.format(n_samples))
        
#         if ax is None:
#             fig = plt.figure(figsize=(5,5))
#             ax = plt.gca()
#         else:
#             pass
            
#         if powernorm:
#             pm = ax.pcolormesh(xedge,yedge,C.transpose(),cmap=cmap,edgecolor=edgecolor,norm=colors.PowerNorm(gamma=0.5,vmin=vmin,vmax=vmax),alpha=alpha)
            
#             if cbar:
#                 cbar = plt.colorbar(pm,ax=ax)
#             else:
#                 cbar = pm 
#         else:
#             pm = ax.pcolormesh(xedge,yedge,C.transpose(),cmap=cmap,vmin=vmin,vmax=vmax,edgecolor=edgecolor,alpha=alpha)
#             if cbar:
#                 cbar = plt.colorbar(pm,ax=ax)
#             else:
#                 cbar = pm 
            
#         return ax,cbar,C
    
#     elif unconditional:
    
#         df = pd.DataFrame({'x':ind1-1,'y':ind2-1,'c':c})
#         if method=='mean':
#             df2 = df.groupby(["x","y"])['c'].sum()
            
#         df3 = df.groupby(["x","y"]).count()
#         df2 = df2.to_frame()
#         df2.insert(1,'Count',df3.values)
#         df = df2.where(df2.Count >= mincnt).dropna()
#         C = np.ones([xedge.shape[0]-1,yedge.shape[0]-1])
#         for i,ii in enumerate(df.index.values):
#             C[ii[0],ii[1]] = df.c.values[i]
                
#         C = C/master_count.values

#         if ax is None:
#             fig = plt.figure(figsize=(5,5))
#             ax = plt.gca()
#         else:
#             pass
        
#         if powernorm:
#             pm = ax.pcolor(xedge,yedge,C.transpose(),cmap=cmap,norm=colors.PowerNorm(gamma=0.5,vmin=vmin,vmax=vmax),alpha=alpha)
#             if cbar:
#                 cbar = plt.colorbar(pm,ax=ax)
#         else:
            
#             pm = ax.pcolor(xedge,yedge,C.transpose(),cmap=cmap,vmin=vmin,vmax=vmax,alpha=alpha)
#             if cbar: 
#                 cbar = plt.colorbar(pm,ax=ax)
        
        
#     else:
#         df = pd.DataFrame({'x':ind1-1,'y':ind2-1,'c':c})
#         if method=='mean':
#             df2 = df.groupby(["x","y"])['c'].mean()
#         elif method=='std':
#             df2 = df.groupby(["x","y"])['c'].std()
#         elif method=='median':
#             df2 = df.groupby(["x","y"])['c'].median()
#         elif method=='qunatile':
#             if quantile is None:
#                 print('No quantile given, defaulting to median')
#                 quantile = 0.5
#             else:
#                 pass
#             df2 = df.groupby(["x","y"])['c'].apply(percentile(quantile*100))
            
            
#         df3 = df.groupby(["x","y"]).count()
#         df2 = df2.to_frame()
#         df2.insert(1,'Count',df3.values)
#         df = df2.where(df2.Count >= mincnt).dropna()
#         C = np.ones([xedge.shape[0]-1,yedge.shape[0]-1])*-9999
#         for i,ii in enumerate(df.index.values):
#             C[ii[0],ii[1]] = df.c.values[i]

#         C = np.ma.masked_where(C == -9999,C)

#         if ax is None:
#             fig = plt.figure(figsize=(5,5))
#             ax = plt.gca()
#         else:
#             pass
        
#         if powernorm:
#             pm = ax.pcolor(xedge,yedge,C.transpose(),cmap=cmap,vmin=vmin,vmax=vmax,norm=colors.PowerNorm(gamma=0.5),alpha=alpha)
#             if cbar:
#                 cbar = plt.colorbar(pm,ax=ax)
#             else:
#                 cbar = pm
#         else:
            
#             pm = ax.pcolor(xedge,yedge,C.transpose(),cmap=cmap,vmin=vmin,vmax=vmax,alpha=alpha)
#             if cbar: 
#                 cbar = plt.colorbar(pm,ax=ax)
#             else:
#                 cbar = pm 
            
#     return ax,cbar,C

# def make_colorbar(ax,vmin,vmax,cmap):
#     cNorm  = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
#     scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cmap)
#     cb1 = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap,
#                               norm=cNorm,
#                               orientation='vertical',extend='max')
#     return cb1

# import copy 
# import pandas as pd
# import matplotlib.colors as colors
# import matplotlib.cm as cmx
# import matplotlib 
# import matplotlib.patheffects as path_effects
# import cmocean
# import matplotlib.pyplot as plt

# #plot parameters that I personally like, feel free to make these your own.
# matplotlib.rcParams['axes.facecolor'] = [0.9,0.9,0.9]
# matplotlib.rcParams['axes.labelsize'] = 14
# matplotlib.rcParams['axes.titlesize'] = 14
# matplotlib.rcParams['xtick.labelsize'] = 12
# matplotlib.rcParams['ytick.labelsize'] = 12
# matplotlib.rcParams['legend.fontsize'] = 12
# matplotlib.rcParams['legend.facecolor'] = 'w'
# matplotlib.rcParams['savefig.transparent'] = False


# files = glob.glob('/PATH/TO/DATA/FROM/BIT/ONE/TESTS_*.nc')
# files.sort()

# for i,file in enumerate(files):
#     ds = xr.open_dataset(file)

#     print(ds)

#     y_np = ds.y_true.values
#     y_out = ds['median'].values

#     #ml pred 
#     x = copy.deepcopy(y_out.ravel())
#     #'truth'
#     y = copy.deepcopy(y_np.ravel())

#     xbin = np.linspace(0,50)
#     ybin = np.linspace(0,50)

#     fig_dummy = plt.figure()
#     ax = plt.gca()
#     #this takes a long time with all the data.... 
#     # ax.scatter(x,y,s=1)
#     ax,cbar,C = boxbin(x,y,xbin,ybin,ax=ax,powernorm=False,vmax=1e5,cmap='Spectral_r',mincnt=0)

#     plt.close(fig_dummy)
    
#     if i == 0:
#         C_accum = copy.deepcopy(np.asarray(C))

#         dropers = ~np.isnan(x)
#         x = x[dropers]
#         y = y[dropers]
#         x_accum = copy.deepcopy(x.astype(np.float16))
#         y_accum = copy.deepcopy(y.astype(np.float16))
#     else:
#         C_accum += np.asarray(C)

#         dropers = ~np.isnan(x)
#         x = x[dropers]
#         y = y[dropers]

#         x_accum = np.append(x_accum,x.astype(np.float16))
#         y_accum = np.append(y_accum,y.astype(np.float16))

#     ds.close()
#     del ds,x,y,C,y_np,y_out




# plt.figure(figsize=(6,5),facecolor='w')

# ax = plt.gca()

# ds = xr.DataArray(C_accum,dims=['x','y'],name='C_accum').to_dataset()

# ds.to_netcdf('/PATH/3dmodel_counts_test.nc')

# C_accum[C_accum <= 0] = np.nan

# # pm = ax.pcolor(xbin,ybin,C_accum.transpose(),cmap='Spectral_r',norm=colors.PowerNorm(gamma=0.5,vmin=0,vmax=0.25e6))
# pm = ax.pcolor(xbin,ybin,np.log10(C_accum.transpose()),cmap='turbo',vmin=2,vmax=6)
# cbar = plt.colorbar(pm,ax=ax)
# cbar.set_label(r'Number of samples in bin')

# plt.plot([0,35],[0,35],'-k')
# plt.xlim([0,35])
# plt.ylim([0,35])

# plt.xlabel('ML W, [$m \ s^{-1}$]')
# plt.ylabel('WoFS W , [$m \ s^{-1}$]')

# #need to make this savestring match the model name 
# plt.savefig('/PATH/scatter_test_3d.png',dpi=300)

# #get 'usual' R^2 
# import scipy.stats as stats
# slope, intercept, r_value, p_value, std_err = stats.linregress(x_accum,y_accum)
# print('R^2: {}'.format(r_value**2))

# MAE = tf.keras.metrics.MeanAbsoluteError()
# MAE.update_state(y_accum,x_accum)
# print('MAE: {}'.format(MAE.result().numpy()))

# RMSE = tf.keras.metrics.MeanSquaredError()
# RMSE.update_state(y_accum,x_accum)
# print('RMSE: {}'.format(np.sqrt(RMSE.result().numpy())))

# #
# print('Conditional Stats')
# for i in [1,5,10,15,20]:
#     print(i)
#     idx = np.where(y_accum >= i)[0]
#     print(len(idx))
#     y_tmp = y_accum[idx]
#     x_tmp = x_accum[idx]
#     RMSE.reset_state()
#     RMSE.update_state(y_tmp,x_tmp)
#     print('RMSE: {}'.format(np.sqrt(RMSE.result().numpy())))

#     MAE.reset_state()
#     MAE.update_state(y_tmp,x_tmp)
#     print('MAE: {}'.format(MAE.result().numpy()))
