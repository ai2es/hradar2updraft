"""

This script was used to create the large database of WoFS data. This is the first step at grabbing one patch from each run of WoFS

Note: all paths removed for security

""" 

from netCDF4 import Dataset
from wrf import getvar,interplevel
import numpy as np 
import copy
import xarray as xr
import pandas as pd
from pathlib import Path
import gc 
import sys 

def patcher_par(filename):

    savepath = '#'
    patchsize = 128
    n_patches = 1
    try:
        wrfin = Dataset(filename)
    except:
        print('boken file: {}'.format(filename))
        return
        
    #get current time 
    current_dtime = pd.to_datetime(''.join(np.asarray(wrfin['Times'][:],dtype='str').squeeze().tolist()),
                                  format='%Y-%m-%d_%H:%M:%S')
    #get inital time 
    wofs_starttime = pd.to_datetime(wrfin.START_DATE,format='%Y-%m-%d_%H:%M:%S')
    #calc time difference 
    valid_time_diff  = ((current_dtime - wofs_starttime).total_seconds()/60)
    
    #get altitudes of model levels
    height = getvar(wrfin, "height_agl",units='m')
    #get Z
    Z = getvar(wrfin, "REFL_10CM")
    #interpolate it to common gridrad levels (adjusted for just 24 heights on 12/16/22 by RJC)
    gridrad_levels = np.array([ 0.5,  1. ,  1.5,  2. ,  2.5,  3. ,  3.5,  4. ,  4.5,  5. ,  5.5,  6. ,
        6.5,  7. ,  8. ,  9., 10., 11., 12., 13., 14., 15., 16., 17.,])*1000

    #moved back to the ful amount, because we are running with the 2D model for now... 
    # gridrad_levels = np.array([ 0.5,  1. ,  1.5,  2. ,  2.5,  3. ,  3.5,  4. ,  4.5,  5. ,  5.5,  6. ,
                                # 6.5,  7. ,  8. ,  9. , 10. , 11. , 12. , 13. , 14. , 15. , 16. , 17., 
                                # 18. , 19. , 20. , 21. , 22. ])*1000
    
    Z_agl = interplevel(Z,height, gridrad_levels)
    #swap dim so we can have channels last 
    Z_agl = np.moveaxis(Z_agl.values,[0],[-1])

    #get W max (labels)
    W = getvar(wrfin, "W_UP_MAX")
    W = W.values

    # Define a patches array to be 32 by 32 for 100 values
    w_patches = np.zeros([n_patches,patchsize,patchsize,1],dtype=np.float32)
    z_patches = np.zeros([n_patches,patchsize,patchsize,len(gridrad_levels)],dtype=np.float32)
    
    #define quantiles 
    q = [0,10,25,50,75,90,100]
    w_patch_q = np.zeros([n_patches,len(q)],dtype=np.float32)
    z_patch_q = np.zeros([n_patches,len(q)],dtype=np.float32)
    
    for idx in np.arange(0,n_patches):
        # Choose a starting point x and y value
        x = np.random.choice(np.arange(0,W.shape[0]-patchsize),size=1,replace=True)[0]
        y = np.random.choice(np.arange(0,W.shape[1]-patchsize),size=1,replace=True)[0]
        # Create a randomized patch
        # From the x coordinate through the x coordinate through 32 (swaped x-y here for easy plotting 
        w_patch = copy.deepcopy(W[y:y+patchsize,x:x+patchsize])
        z_patch = copy.deepcopy(Z_agl[y:y+patchsize,x:x+patchsize,:])
        #fill arrays
        w_patches[idx,:,:,:] = copy.deepcopy(w_patch[:,:,np.newaxis])
        z_patches[idx,:,:,:] = copy.deepcopy(z_patch)
        #calc quantiles 
        w_patch_q[idx,:] = np.percentile(w_patch.ravel(),q)
        z_patch_q[idx,:] = np.percentile(np.nanmax(z_patch,axis=-1).ravel(),q)

        
    #make dataset 
    da_w = xr.DataArray(w_patches,dims=['n_samples','nx','ny','nlabel'],
                        name='w_patch',attrs={'units':'m/s','long_name':'patch of max updraft speed'})
    da_w_q=xr.DataArray(w_patch_q,dims=['n_samples','n_quantile'],
                        name='w_quantiles',attrs={'units':'m/s','long_name':'statistics of the patch max updraft speed'})
    da_z = xr.DataArray(z_patches[...,np.newaxis],dims=['n_samples','nx','ny','nz','nchannel'],
                        name='z_patch',attrs={'units':'dBZ','long_name':'patch of reflectivity'})
    da_z_q=xr.DataArray(z_patch_q,dims=['n_samples','n_quantile'],
                        name='z_quantiles',attrs={'units':'dBZ','long_name':'statistics of the comp Z patch'})
    
    da_q = xr.DataArray(q,dims='n_quantile',name='quantiles')

    #add in the file name for easy grabbing later. Note this will only work if n_samples is == 1 
    da_f = xr.DataArray([filename],dims='n_samples',name='filename')
    
    ds = xr.merge([da_w,da_w_q,da_z,da_z_q,da_q,da_f])
    
    
    #build save string 
    validtime = str(np.char.rjust(str(int(valid_time_diff)),width=3,fillchar='0'))
    savestr = 'wofs_updraft_patches_' +validtime + '_' + filename[-19:] + '.nc'

    #get member name to split off of
    MEM = filename.split('/')[-2]
    #get yearmonthday to split
    YMD = filename.split('/')[-4]

    #make dir if it doesnt exist 
    Path(savepath + YMD + '/' + MEM + '/').mkdir(parents=True, exist_ok=True)

    #save it out 
    ds.to_netcdf(savepath + YMD + '/' + MEM + '/' + savestr)

    #close dataset 
    ds.close()

    #clean up some things 
    del ds, wrfin,height,Z,Z_agl,W,w_patches,z_patches,w_patch_q,z_patch_q

    del da_w,da_w_q,da_z,da_z_q,da_q

    gc.collect()

    return

#get wofs filepaths  
import glob 

#2018 Validation set 
# files = glob.glob('#')
# files.sort()

#2019 Training set, minus the one day example, which happens to be the day with it still labeled wrfout
# files = glob.glob('#')
# files.sort()

#2020 
# files = glob.glob('#')
# files.sort()

# print('total files:', len(files))

#determine data splits for this job 

# Arguments passed
input_array_id = int(sys.argv[1])
total_array_jobs = float(sys.argv[2])

#determine steps 
step = np.floor(len(files)/total_array_jobs)

#build idx array
left = np.arange(0,len(files),step,dtype=int)
right = np.arange(0+step,len(files)+step,step,dtype=int)

print(left,right,step)

print('This split. ID:{},START_ID:{},END_ID:{}'.format(input_array_id,left[input_array_id],right[input_array_id]))

#del corrupt files 
# corrupt_set = ['/ourdisk/hpc/ai2es/wofs/2018/20180413/0300/ENS_MEM_10/wrfout_d01_2018-04-14_05:00:00',]
# corrupt_set = ['/ourdisk/hpc/ai2es/wofs/2021/20210407/0300/ENS_MEM_02/wrfwof_d01_2021-04-08_03:40:00']
# files = np.setxor1d(files,corrupt_set).tolist()

#choose a random subset here, the total number of files is insane. (aim for 40,000 but in reality we will only keep 25000)
# print(len(files))
# files = np.random.choice(files,size=40000,replace=False)


import tqdm 

#loop over all files 
# for f in tqdm.tqdm(files):
#     patcher(f,'/scratch/randychase/wofspatches/',32,10)

# parallel loop 
import multiprocessing as mp
pool = mp.Pool(processes=8)
for _ in tqdm.tqdm(pool.imap_unordered(patcher_par,files[left[input_array_id]:right[input_array_id]]), total=len(files[left[input_array_id]:right[input_array_id]])):
    pass

#load and curate 

# import glob 
# files = glob.glob('#')

# ds = xr.open_mfdataset(files,combine='nested',concat_dim='n_samples')

# wmax = ds.w_quantiles[:,-1].values

# #must have m/s scale data in it 
# idx = np.where(wmax > 10)[0]

# ds = ds.isel(n_samples=idx)

# print(ds)

# ds.to_netcdf('#')