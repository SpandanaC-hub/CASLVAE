import os 
os.chdir("./CA_VAE/code")

import torch
from train_utils import train_model
from CA_model import CAVAE_gumbel
from torch.utils.data import Dataset, DataLoader
from data_utils import CreateDataset, train_test_split_unpaired
import matplotlib.pyplot as plt
import pandas as pd


if torch.cuda.is_available():
    DEVICE='cuda'
else:
    DEVICE='cpu'
seed=42
torch.cuda.manual_seed(seed)
torch.manual_seed(42)
torch.cuda.manual_seed_all(seed)

# load training data
data = pd.read_csv('x/y/z')
# select features
# Include ROIs of choice
df_muse = pd.read_csv('a/b/c')
col_muse_ind = df_muse[df_muse.ROI_LEVEL=='SINGLE'].ROI_INDEX.tolist()
#col_muse_ind = list(set(col_muse_ind)-set([4,11,49,50,51,52,46,63,64]))# exclude ventricle and csf rois
col_muse_ind = list(set(col_muse_ind)-set([46,63,64]))# include ventricles and exclude csf rois
print('num features', len(col_muse_ind))
# define brain features and covariates for mapping
covariates_list = ['Sex','Age', 'DLICV'] # selected covariates
brain_measures_list = ['_'.join(['H_MUSE_Volume', str(j)])  for j  in  col_muse_ind]

# define model parameters
opt={'nfeatures':len(brain_measures_list), # number of ROIs
            'nLatent_z':15,
            'nLatent_s_cont':2,
            'nLatent_s_cat':1,
            'ncategories':4,
            'mmd_lambda':0,# hyperparametr to control mmd penalty
            'kl_scat_lambda':1,# hyperparametr to control kl divergence of scat 
            'kl_scont_lambda':1,# hyperparametr to control kl divergence of scont for x input
            'kl_sconty_lambda':1,# hyperparametr to control kl divergence of scont for y input
            'kl_z_lambda':1, # hyperparametr to control kl divergence of common z latent
            'norm_salienty_lambda':0, # hyperparametr to control L1-norm penalty over salient reconstruction of y
            'consistency_lambda':0, # hyperparametr to control consistency penalty
            'bias':True, # include bias in linear layers
            'lr':0.001, # learning rate of G, for D it will be lr/5
            'beta1':0.5,  #beta1 hyperparameter in Adam optimizer
            'DEVICE':DEVICE, # use 'cpu' or 'cuda'
            'epochs':100, # total number of epochs
            'decay_epochs': 1, # epoch number when learning rate decay should begin
            'scheduler': False, #True to set learning rate scheduler
            'batch_norm': False, # Batch Norm layers in G and D
            'dropout': False, # dropout layers in G and D
            'path':'./',# path of directory to save model
            'batch_size':128
            }
            
train_dataset=data.copy()
train_dataset.label.replace({'Healthy':'Background'},inplace=True) # select "Background" group
covariate_correct = False
covariates_list = [] # if empty, does not return any covariates
# get train  and cross validation data
train_data, train_label, train_covariate, test_data, test_label, test_covariate, preprocess_fn = train_test_split_unpaired(train_dataset,feature_cols=brain_measures_list, label_col='label', covariate_cols=covariates_list,test_size=0.3,random_state=0,covariate_correct=covariate_correct)

data_params = {
'data_std':preprocess_fn.data_mean,
'data_mean':preprocess_fn.data_std,
'cov_min': preprocess_fn.cov_min,
'cov_max': preprocess_fn.cov_max,
'cov_beta': preprocess_fn.cov_beta,
'covariate_correct': preprocess_fn.covariate_correct,
'feature_names': brain_measures_list,
'covariate_names': covariates_list
}

#define training dataset 
train_dataset = CreateDataset(train_data,train_label)
train_dataloader = DataLoader(train_dataset,batch_size=opt['batch_size'], shuffle=True)
#define cross-val dataset 
cv_dataset = CreateDataset(test_data,test_label)

# get full data if evalauting
train_data_y = torch.from_numpy(train_dataset.data_bg).to(opt['DEVICE']).type(torch.float32)
train_data_x = torch.from_numpy(train_dataset.data_tgt).to(opt['DEVICE']).type(torch.float32)

cv_data_y = torch.from_numpy(cv_dataset.data_bg).to(opt['DEVICE']).type(torch.float32)
cv_data_x =  torch.from_numpy(cv_dataset.data_tgt).to(opt['DEVICE']).type(torch.float32)


# define model
model_ca = CAVAE_gumbel()
model_ca.create(opt=opt)
model.set_preprocessing_params(data_params)
model_ca, clusteringM, loss_list, converged = train_model(model_ca, opt, train_dataloader, train_data_y, train_data_x, cv_data_y, cv_data_x)

