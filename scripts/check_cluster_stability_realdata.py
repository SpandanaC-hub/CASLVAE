
# contrastive VAE

import os, fnmatch, time, datetime, pickle, argparse, scipy, gzip, pickle5
import sys
curr=os.getcwd()
sys.path.insert(0,os.path.join(curr,'code'))
import pandas as pd, numpy as np, random
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from cluster_utils import crossval_cluster_stability, pairwise_ari

sys.path.insert(0, 'C:/Anaconda3/envs/normative/lib/site-packages')

__author__ = "Spandana Chintapalli"
__copyright__ = "Copyright 2023-2024 The CBICA & SBIA Lab"
__credits__ = ["Spandana Chintapalli"]
__license__ = "See LICENSE file"
__version__ = "0.1.0"
__maintainer__ = "Spandana Chintapalli"
__email__ = "saispandanachintapalli@gmail.com"
__status__ = "Development"

if torch.cuda.is_available():
    DEVICE='cuda'
else:
    DEVICE='cpu'

seed=42
torch.cuda.manual_seed(seed)
torch.manual_seed(42)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
          
#load training data
severity = 'real'
# load istaging
with gzip.open('/cbica/projects/ISTAGING/Pipelines/ISTAGING_Data_Consolidation_2020/v2.0/istaging.pkl.gz','r') as f:
    df_istag = pickle5.load(f)#('c:/Users/ChintapS/Documents/FITBIR_harmonize/May_2023_fitbir_istaging/istaging.pkl')
    

df = df_istag[df_istag.H_MUSE_Volume_47.notnull()].reset_index(drop=True) # select imaging samples
df = df.drop_duplicates('PTID').reset_index(drop=True) # select one sample per ID

# select 45 and older
df= df[(df.Age>=45) & (df.Age<=80)].reset_index(drop=True) # select older adults

df_cn = df[df['Diagnosis_nearest_2.0']=='CN'].reset_index(drop=True)
df_pt = df[df['Diagnosis_nearest_2.0']!='CN'].reset_index(drop=True)
print('Diagnosis', df_pt['Diagnosis_nearest_2.0'].unique())
#df_pt = df_pt[df_pt['Diagnosis_nearest_2.0'].isin(['AD','MCI'])].reset_index(drop=True) # select 
#df_pt = df_pt[~(df_pt.MMSE>29)].reset_index(drop=True) 
# whether to use adni or not
df_cn=df_cn[~df_cn.Study.isin(['ADNI'])].reset_index(drop=True)#'BIOCARD', 'BLSA','WHIMS', 'WRAP','SHIP','PENN',AIBL', 'OASIS'
df_pt=df_pt[~df_pt.Study.isin(['ADNI'])].reset_index(drop=True)

df_cn = df_cn[~df_cn['Diagnosis_nearest_2.0'].isnull()].reset_index(drop=True)
df_pt = df_pt[~df_pt['Diagnosis_nearest_2.0'].isnull()].reset_index(drop=True)
# whether to use adni or not
#df_cn=df_cn[df_cn.Study.isin(['OASIS','PENN'])].reset_index(drop=True)#'BIOCARD', 'BLSA','WHIMS', 'WRAP','SHIP','PENN',AIBL', 'OASIS'
#df_pt=df_pt[df_pt.Study.isin(['OASIS','PENN'])].reset_index(drop=True)
#when using ADNI
#df_cn=df_cn[df_cn.Phase.isin(['ADNI2','ADNIGO'])].reset_index(drop=True) 
#df_pt=df_pt[df_pt.Phase.isin(['ADNI2', 'ADNIGO'])].reset_index(drop=True)

print('diagnosis labels', df_pt['Diagnosis_nearest_2.0'].unique(),'CN sample size', df_cn.shape,'PT sample size',df_pt.shape)
df_cn.DLICV = df_cn.DLICV_baseline
df_pt.DLICV = df_pt.DLICV_baseline


data = pd.concat([df_cn,df_pt],axis=0).reset_index(drop=True)
data=data.rename(columns={'Diagnosis_nearest_2.0':'label'})
print('Trainind data labels', data.label.value_counts())
print('Study counts', data.groupby('Study').label.value_counts())
# Include ROIs of choice
df_muse = pd.read_csv('/cbica/home/chintaps/GAN_normative/HybridGAN_snapshot_mini_15Dec2024/data/MUSE_ROI_Dictionary.csv')
col_muse_ind = df_muse[df_muse.ROI_LEVEL=='SINGLE'].ROI_INDEX.tolist()
col_muse_ind = list(set(col_muse_ind)-set([46,63,64]))# include ventricles and exclude csf rois

# define brain features and covariates for mapping
covariates_list = ['Sex','Age', 'DLICV'] # if empty, does not return any covariates
brain_measures_list = ['_'.join(['H_MUSE_Volume', str(j)])  for j  in  col_muse_ind]

# Define Datasets
dataset=data.copy()
dataset.label.replace({'CN':'Background'},inplace=True) # select "Background" group
dataset.Sex = dataset.Sex.replace({'F':1,'M':0}) # Recode Sex
# Set to True if you want to correct for covariates
covariate_correct = True

if not covariate_correct:
  covariates_list=[]
  # define model options, set mmd lambda and consistency lamda as lower values 
  opt={'nfeatures':len(brain_measures_list), # number of ROIs
        'nLatent_z':15,
        'nLatent_s_cont':2,
        'nLatent_s_cat':1,
        'ncategories':4,
        'mmd_lambda':100,# hyperparameter to control mmd penalty
        'kl_scat_lambda':1,# hyperparameter to control kl on s cat for x  
        'kl_scont_lambda':1,# hyperparameter to control kl on s cont for x
        'kl_sconty_lambda':1,# hyperparameter to control penalty on s cont for y 
        'kl_z_lambda':1, # hyperparameter to control kl on z for x and y  
        'consistency_lambda':10, # hyperparameter to control latent consistency of z
        'norm_salienty_lambda': 0, # hyperparameter to control L1 norm penalty on salient reconstruction for y
        'entropy_lambda':1,
        'lr':0.001, # learning rate of G, for D it will be lr/5
        'beta1':0.5,  #beta1 hyperparameter in Adam optimizer
        'DEVICE':DEVICE, # use 'cpu' or 'cuda'
        'epochs':500, # max training epochs
        'decay_epochs': 1, # epoch number when learning rate decay should begin
        'scheduler': False, #True to set learning rate scheduler
        'path':f'/cbica/home/chintaps/CA_VAE/results/severity{severity}_clusterstability/CASLVAE_2dec_wcov_allistagingwoadnirerun/',# path of directory to save model
        'batch_size':128
        }
else:
  # define model options
  opt={'nfeatures':len(brain_measures_list), # number of ROIs
        'nLatent_z':15,
        'nLatent_s_cont':2,
        'nLatent_s_cat':1,
        'ncategories':4,
        'mmd_lambda':100,# hyperparameter to control mmd penalty
        'kl_scat_lambda':1,# hyperparameter to control kl on s cat for x  
        'kl_scont_lambda':1,# hyperparameter to control kl on s cont for x
        'kl_sconty_lambda':1,# hyperparameter to control penalty on s cont for y 
        'kl_z_lambda':1, # hyperparameter to control kl on z for x and y  
        'consistency_lambda':10, # hyperparameter to control latent consistency of z
        'norm_salienty_lambda': 0, # hyperparameter to control L1 norm penalty on salient reconstruction for y
        'entropy_lambda':1,
        'lr':0.001, # learning rate of G, for D it will be lr/5
        'beta1':0.5,  #beta1 hyperparameter in Adam optimizer
        'DEVICE':DEVICE, # use 'cpu' or 'cuda'
        'epochs':500, # max training epochs
        'decay_epochs': 1, # epoch number when learning rate decay should begin
        'scheduler': False, #True to set learning rate scheduler
        'path':f'/cbica/home/chintaps/CA_VAE/results/severity{severity}_clusterstability/CASLVAE_2dec_wocov_allistagingwoadnirerun/',# path of directory to save model
        'batch_size':128
        }
os.makedirs(opt['path'], exist_ok=True)

M_list = [2,3,4,5,6]
clustering_logits = crossval_cluster_stability(dataset.copy(),opt, covariate_correct, brain_measures_list, covariates_list, M_list)

     
pairwise_ari_results = {}
for i, cluster in enumerate(clustering_logits):
    pairwise_ari_results[M_list[i]] =  pairwise_ari(cluster)
    
with open(os.path.join(opt['path'], f'pairwise_ari_dict.pkl'), 'wb') as f:
	pickle.dump(pairwise_ari_results, f, protocol=pickle.HIGHEST_PROTOCOL)

keys = list( pairwise_ari_results.keys())
values = list( pairwise_ari_results.values())
plt.boxplot( values, showmeans=True)
plt.xticks(list(np.arange(1,len(M_list)+1)),M_list)
plt.ylabel('Pairwise ARI 10-fold cross val')
plt.xlabel('Number of Categories M')
plt.savefig(os.path.join(opt['path'],f'pairwiseari.jpg'))
plt.close()

clustering_results={}
clustering_results['logits'] = clustering_logits   
with open(os.path.join(opt['path'], f'clustering_results.pkl'), 'wb') as f:
	pickle.dump(clustering_results, f, protocol=pickle.HIGHEST_PROTOCOL)
