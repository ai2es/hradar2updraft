# Machine Learning Estimate of Storm Updrafts

This repositroy holds the code for the paper titled "Machine Learning Estimation of Maximum Vertical Velocity from Radar" accpeted AIES. A preprint of the paper can be read [here](https://journals.ametsoc.org/view/journals/aies/aop/AIES-D-23-0095.1/AIES-D-23-0095.1.xml). 

The name of this repo comes from the image-to-image translation task of horizontal (h) slices of 3-dimensional radar data to (2) updrafts (i.e., hradar2updraft). This was done in the anticipation of a vertical (v) radar data to updraft (i.e., vradar2updraft) method that might come about for use with the Global Precipitation Measurement (GPM) Dual-frequency Precipitation Radar (DPR) or other vertically pointing radars (e.g., TRMM; AOS etc.)

Note, I am working on some code to get this running with MRMS data, but since this project is currently unfunded, it will take me some time to get the code ironed out. Potentially a summer intern will help me with this in Summer 2024. 

## Getting Started
This assumes you already have git and python. If you are new to python, go ahead and get [mamba](https://mamba.readthedocs.io/en/latest/installation.html). Alternatively, if you wish to not have to install anything see the google colab folder. 

1. In a terminal window, clone this repository to your local machine with the command

   `git clone https://github.com/ai2es/hradar2updraft.git`

2. Within the terminal, go to the top-level directory of your repository with cd 

   `cd hradar2updraft` 

3. Create new virtual enviroment from the environment.yml file.
   Use the command 

   `mamba env create -f environment.yml`

   if you do not have mamba, swap out conda above. Let it install all the needed packages. 

3. Activate the new environment by running 

   `mamba activate hradar2updraft` 

4. If you want to run the model in jupyter, you will need to add this new enviroment to a jupyter kernel. 
   
