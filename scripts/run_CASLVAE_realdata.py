
# contrastive VAE

import os, fnmatch, time, datetime, pickle, argparse, scipy, gzip, pickle5

import sklearn.model_selection
import sys
curr=os.getcwd()
sys.path.insert(0,os.path.join(curr,'code'))
import sklearn
import torch 
import pandas as pd, numpy as np, random
from CASLVAE_model import CASLVAE
from train_utils import train_model
from torch.utils.data import Dataset, DataLoader
from data_utils import CreateDataset, train_test_split_unpaired
import matplotlib.pyplot as plt
import seaborn as sns

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

seed=10#42
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
df= df[(df.Age>=45)].reset_index(drop=True) # select older adults

df_cn = df[df['Diagnosis_nearest_2.0']=='CN'].reset_index(drop=True)
df_pt = df[df['Diagnosis_nearest_2.0']!='CN'].reset_index(drop=True)
#df_pt = df_pt[df_pt['Diagnosis_nearest_2.0'].isin(['AD','MCI','early MCI'])].reset_index(drop=True) # select 

# whether to use adni or not
df_cn=df_cn[~df_cn.Study.isin(['ADNI'])].reset_index(drop=True)#'BIOCARD', 'BLSA','WHIMS', 'WRAP','SHIP','PENN',AIBL', 'OASIS'
df_pt=df_pt[~df_pt.Study.isin(['ADNI'])].reset_index(drop=True)

# whether to use adni or not
#df_cn=df_cn[df_cn.Study.isin(['OASIS','PENN'])].reset_index(drop=True)#'BIOCARD', 'BLSA','WHIMS', 'WRAP','SHIP',
#df_pt=df_pt[df_pt.Study.isin(['OASIS','PENN'])].reset_index(drop=True)#PENN, AIBL

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
        'ncategories':5,
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
        'path':f'/cbica/home/chintaps/CA_VAE/code_test/results/severity{severity}_10folCV_allistaging/CASLVAE_2dec_wcov_allistagingwoadni5cat/',# path of directory to save model
        'batch_size':128
        }
else:
  # define model options
  opt={'nfeatures':len(brain_measures_list), # number of ROIs
        'nLatent_z':15,
        'nLatent_s_cont':2,
        'nLatent_s_cat':1,
        'ncategories':5,
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
        'path':f'/cbica/home/chintaps/CA_VAE/code_test/results/severity{severity}_10folCV_allistaging/CASLVAE_2dec_allistagingwoadni5cat/',# path of directory to save model
        'batch_size':128
        }
os.makedirs(opt['path'], exist_ok=True)
'''
results={}
#10-fold cv
for fold in range(10):
  
  # get train  and cross validation data
  train_data, train_label, train_covariate, test_data, test_label, test_covariate, preprocess_fn = train_test_split_unpaired(dataset.copy(),feature_cols=brain_measures_list, label_col='label', covariate_cols=covariates_list,test_size=0.2,covariate_correct=covariate_correct)
  
  data_params = {
  'data_std':preprocess_fn.data_std,
  'data_mean':preprocess_fn.data_mean,
  'cov_min': preprocess_fn.cov_min,
  'cov_max': preprocess_fn.cov_max,
  'cov_beta': preprocess_fn.cov_beta,
  'covariate_correct': preprocess_fn.covariate_correct,
  'feature_names': brain_measures_list,
  'covariate_names': covariates_list,
  'tau':1
  }

  train_data, cv_data, train_label, cv_label = sklearn.model_selection.train_test_split(train_data, train_label, test_size=0.1)
  #define dataset and dataloader
  train_dataset = CreateDataset(train_data,train_label)
  train_dataloader = DataLoader(train_dataset,batch_size=opt['batch_size'], shuffle=True)
  #define cross-val dataset 
  cv_dataset = CreateDataset(cv_data,cv_label)
  
  # load data for evalauting training
  train_data_y = torch.from_numpy(train_dataset.data_bg).to(opt['DEVICE']).type(torch.float32) # get reference/background
  train_data_x = torch.from_numpy(train_dataset.data_tgt).to(opt['DEVICE']).type(torch.float32) # get target
  
  cv_data_y = torch.from_numpy(cv_dataset.data_bg).to(opt['DEVICE']).type(torch.float32) # get reference
  cv_data_x =  torch.from_numpy(cv_dataset.data_tgt).to(opt['DEVICE']).type(torch.float32) # get target
  # define model and start training
  converged=False
  counter=0
  while not converged:
    model = CASLVAE()
    model.create(opt=opt)
    model.set_preprocessing_params(data_params)
    model, clusteringM, loss_list, converged = train_model(model, opt, train_dataloader, train_data_y, train_data_x, cv_data_y, cv_data_x)
    print('Training has converged:', converged)
    counter+=1
    if counter==3:
      break
  print('Number of training trials:', counter)
  
  scat_prob = scipy.special.softmax(clusteringM[0].squeeze(),axis=-1) # get scat predictions
  
  plt.figure(figsize=(35,10))
  plt.subplot(2,7,1)
  plt.plot(loss_list['recon_x'][0])
  plt.plot(loss_list['recon_x'][1])
  plt.plot(loss_list['recon_y'][0])
  plt.plot(loss_list['recon_y'][1])
  plt.legend(['recon pt','recon eva pt','recon hc','recon eva hc'])
  
  plt.subplot(2,7,2)
  plt.plot(loss_list['w_list_pt'][0])
  plt.plot(loss_list['w_list_pt'][1])
  plt.plot(loss_list['w_list_hc'][0])
  plt.plot(loss_list['w_list_hc'][1])
  plt.legend(['wd pt','wd eva pt','wd hc','wd eva hc'])
  
  plt.subplot(2,7,3)
  plt.plot(loss_list['w_list_pair'][0])
  plt.plot(loss_list['w_list_pair'][1])
  plt.legend(['wd unpaired train','wd unpaired synth'])
  
  plt.subplot(2,7,4)
  plt.plot(loss_list['kl_loss'][0])
  plt.title('kl z')
  plt.subplot(2,7,5)
  plt.plot(loss_list['kl_loss'][1])
  plt.title('kl scat x')
  plt.subplot(2,7,6)
  plt.plot(loss_list['kl_loss'][2])
  plt.title('kl scont x')
  
  plt.subplot(2,7,7)
  plt.plot(loss_list['mmd_loss'])
  plt.title('mmd loss')
  
  plt.subplot(2,7,8)
  plt.plot(loss_list['tau'])
  plt.title('gumbel temperature tau')
  
  plt.subplot(2,7,9)
  plt.plot(loss_list['entropy'])
  plt.title('cross-entropy on scat')
  
  plt.subplot(2,7,10)
  plt.plot(loss_list['norm_reconsalient_y'])
  plt.title('L1 norm of y salient recon')
  
  plt.subplot(2,7,11)
  plt.plot(loss_list['consistency_list'])
  plt.title('latent consistency')
  
  plt.subplot(2,7,12)
  plt.plot(loss_list['kl_loss'][3])
  plt.title('kl scont y penalty')
  
  plt.subplot(2,7,13)
  plt.plot(loss_list['kl_loss'][4])
  plt.title('kl scontprior y penalty')
  
  plt.subplot(2,7,14)
  sns.scatterplot(x = scat_prob[:,-2], y=scat_prob[:,-1], hue=cv_dataset.tgt_labels)
  sns.scatterplot(x=scat_prob[:,0], y=scat_prob[:,0], hue=cv_dataset.tgt_labels)
  
  plt.xlabel(f'scat {opt["ncategories"]}')
  plt.ylabel(f'scat {opt["ncategories"]+1}')
  plt.title('scat probabilities')
  plt.savefig(os.path.join(opt['path'],f'monitor_loss_{fold}.jpg'))
  plt.close()
  
  
  # evaluate on held-out
  test_data = torch.from_numpy(test_data).to(opt['DEVICE']).type(torch.float32) # get data
  
  model.G_forward.eval()
  with torch.no_grad():
    output_infer = model.G_forward(test_data,test_data,tau=model.tau)

    cat_probs = scipy.special.softmax(output_infer['logits_x'].detach().to('cpu').numpy().squeeze(),axis=-1)
    
    x_recon = output_infer['x_recon'].detach().to('cpu').numpy()

    
    results[f'{fold}']={}
    results[f'{fold}']['data_x'] = test_data.detach().to('cpu').numpy()
    results[f'{fold}']['recon_x'] = x_recon
    results[f'{fold}']['label_x'] = test_label 
    results[f'{fold}']['cat_probs_x'] = cat_probs
    

with open(os.path.join(opt['path'], f'results_{severity}_dict.pkl'), 'wb') as f:
	pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
'''
indtrain,indtest=sklearn.model_selection.train_test_split(dataset.index.tolist(),test_size=0.3)#test_df.index.tolist()
dataset.loc[indtest,:].to_csv(os.path.join(opt['path'],'test_split.csv'),index=False)
dataset = dataset.loc[indtrain,:].reset_index(drop=True)
#Use all data to train # not 10 fold but 1 fold CV
for fold in range(1):
  # get train  and cross validation data
  train_data, train_label, train_covariate, test_data, test_label, test_covariate, preprocess_fn = train_test_split_unpaired(dataset.copy(),feature_cols=brain_measures_list, label_col='label', covariate_cols=covariates_list,test_size=0.01,covariate_correct=covariate_correct)

  data_params = {
  'data_std':preprocess_fn.data_std,
  'data_mean':preprocess_fn.data_mean,
  'cov_min': preprocess_fn.cov_min,
  'cov_max': preprocess_fn.cov_max,
  'cov_beta': preprocess_fn.cov_beta,
  'covariate_correct': preprocess_fn.covariate_correct,
  'feature_names': brain_measures_list,
  'covariate_names': covariates_list,
  'tau':1
  }

  #define dataset and dataloader
  train_dataset = CreateDataset(train_data,train_label)
  train_dataloader = DataLoader(train_dataset,batch_size=opt['batch_size'], shuffle=True)
  #define cross-val dataset 
  cv_dataset = CreateDataset(test_data,test_label)
  
  # load data for evalauting training
  train_data_y = torch.from_numpy(train_dataset.data_bg).to(opt['DEVICE']).type(torch.float32) # get reference/background
  train_data_x = torch.from_numpy(train_dataset.data_tgt).to(opt['DEVICE']).type(torch.float32) # get target
  
  cv_data_y = torch.from_numpy(cv_dataset.data_bg).to(opt['DEVICE']).type(torch.float32) # get reference
  cv_data_x =  torch.from_numpy(cv_dataset.data_tgt).to(opt['DEVICE']).type(torch.float32) # get target
  # define model and start training
  converged=False
  counter=0
  while not converged:
    model = CASLVAE()
    model.create(opt=opt)
    model.set_preprocessing_params(data_params)
    model, clusteringM, loss_list, converged = train_model(model, opt, train_dataloader, train_data_y, train_data_x, cv_data_y, cv_data_x)
    print('Training has converged:', converged)
    counter+=1
    if counter==3:
      break
  print('Number of training trials:', counter)
  
  scat_prob = scipy.special.softmax(clusteringM[0].squeeze(),axis=-1) # get scat predictions
  
  plt.figure(figsize=(35,10))
  plt.subplot(2,7,1)
  plt.plot(loss_list['recon_x'][0])
  plt.plot(loss_list['recon_x'][1])
  plt.plot(loss_list['recon_y'][0])
  plt.plot(loss_list['recon_y'][1])
  plt.legend(['recon pt','recon eva pt','recon hc','recon eva hc'])
  
  plt.subplot(2,7,2)
  plt.plot(loss_list['w_list_pt'][0])
  plt.plot(loss_list['w_list_pt'][1])
  plt.plot(loss_list['w_list_hc'][0])
  plt.plot(loss_list['w_list_hc'][1])
  plt.legend(['wd pt','wd eva pt','wd hc','wd eva hc'])
  
  plt.subplot(2,7,3)
  plt.plot(loss_list['w_list_pair'][0])
  plt.plot(loss_list['w_list_pair'][1])
  plt.legend(['wd unpaired train','wd unpaired synth'])
  
  plt.subplot(2,7,4)
  plt.plot(loss_list['kl_loss'][0])
  plt.title('kl z')
  plt.subplot(2,7,5)
  plt.plot(loss_list['kl_loss'][1])
  plt.title('kl scat x')
  plt.subplot(2,7,6)
  plt.plot(loss_list['kl_loss'][2])
  plt.title('kl scont x')
  
  plt.subplot(2,7,7)
  plt.plot(loss_list['mmd_loss'])
  plt.title('mmd loss')
  
  plt.subplot(2,7,8)
  plt.plot(loss_list['tau'])
  plt.title('gumbel temperature tau')
  
  plt.subplot(2,7,9)
  plt.plot(loss_list['entropy'])
  plt.title('cross-entropy on scat')
  
  plt.subplot(2,7,10)
  plt.plot(loss_list['norm_reconsalient_y'])
  plt.title('L1 norm of y salient recon')
  
  plt.subplot(2,7,11)
  plt.plot(loss_list['consistency_list'])
  plt.title('latent consistency')
  
  plt.subplot(2,7,12)
  plt.plot(loss_list['kl_loss'][3])
  plt.title('kl scont y penalty')
  
  plt.subplot(2,7,13)
  plt.plot(loss_list['kl_loss'][4])
  plt.title('kl scontprior y penalty')
  
  plt.subplot(2,7,14)
  sns.scatterplot(x = scat_prob[:,-2], y=scat_prob[:,-1], hue=cv_dataset.tgt_labels)
  sns.scatterplot(x=scat_prob[:,0], y=scat_prob[:,0], hue=cv_dataset.tgt_labels)
  
  plt.xlabel(f'scat {opt["ncategories"]}')
  plt.ylabel(f'scat {opt["ncategories"]+1}')
  plt.title('scat probabilities')
  plt.savefig(os.path.join(opt['path'],f'monitor_loss_{fold}.jpg'))
  plt.close()

with open(os.path.join(opt['path'], f'dataparams_{severity}_dict.pkl'), 'wb') as f:
  pickle.dump(data_params, f, protocol=pickle.HIGHEST_PROTOCOL)