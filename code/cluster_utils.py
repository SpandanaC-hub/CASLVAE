import sklearn
import os, pickle
import numpy as np
from sklearn.metrics import  adjusted_rand_score
from CASLVAE_model import CASLVAE
import torch
from train_utils import train_model
from torch.utils.data import Dataset, DataLoader
from data_utils import CreateDataset, get_covariatecorrection_params, apply_covariate_correction, get_normalization_params, apply_normalization


# compute all pairwise aris
def pairwise_ari(clustering_logits):
    all_ari = []
    n = len(clustering_logits)
    for i in range(n):
        for j in range(i+1, n):
            # ARI needs same number of samples ? compare only overlap
            m = min(len(clustering_logits[i]), len(clustering_logits[j]))
            ari = adjusted_rand_score(np.argmax(clustering_logits[i][:m],axis=1), np.argmax(clustering_logits[j][:m],axis=1))
            all_ari.append(ari)
    return all_ari
            
            
# fucntion to measure cluster stability 
def crossval_cluster_stability(data, opt, covariate_correct=False, brain_measures_list=[], covariates_list=[], M_list=[2]):
    #kf=sklearn.model_selection.KFold(n_splits=10, shuffle=True, random_state=42) # if using K-fold
    kf = sklearn.model_selection.ShuffleSplit(n_splits=10, train_size=0.8, random_state=42) 
    clustering = [] # predicted logits , list of lists, each list is 10-CV results for selected M
    # get train  and cross validation data
    if covariate_correct:
        cov_min,cov_max,cov_beta = get_covariatecorrection_params(data.loc[:,brain_measures_list].to_numpy(),data.loc[:,covariates_list].to_numpy(),data.loc[:,'label'].to_numpy())
        params = {'cov_min':cov_min,'cov_max':cov_max,'cov_beta':cov_beta}
        data_features = apply_covariate_correction(data.loc[:,brain_measures_list].to_numpy(),data.loc[:,covariates_list].to_numpy(),params)
        data.loc[:,brain_measures_list] = data_features
    else:
        params = {'cov_min':None,'cov_max':None, 'cov_beta':None}
    params['feature_names'] = brain_measures_list
    params['covariate_names'] = covariates_list
    params['covariate_correct']=covariate_correct
    params['tau']=1
    for M in M_list:
        opt['ncategories'] = M
        clusteringM = [] # save logits in each CV fold for M clusters
        for fold, split in enumerate(kf.split(data)):
            print('Fold no.', fold)
            train_split, test_split = split

            # get features
            #train
            train_data= data.loc[train_split,brain_measures_list].to_numpy()
            train_label = data.loc[train_split,'label'].to_numpy()
            train_mean, train_std = get_normalization_params(train_data, train_label)
            train_data = apply_normalization(train_data, mean = train_mean, std = train_std)
            params['data_mean']=train_mean
            params['data_std']=train_std

            #cross-val for clustering use all PT data
            cv_data = apply_normalization(data.loc[:,brain_measures_list].to_numpy(), mean = train_mean, std = train_std)
            cv_label = data.loc[:,'label'].to_numpy()

            # define training dataset 
            train_dataset = CreateDataset(train_data,train_label)
            train_loader = DataLoader(train_dataset,batch_size=opt['batch_size'], shuffle=True)
            # define cross-val dataset 
            cv_dataset = CreateDataset(cv_data,cv_label)

            # load data 
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
                model.set_preprocessing_params(params)
                model, cluster_logits, loss_list, converged = train_model(model, opt, train_loader, train_data_y, train_data_x, cv_data_y, cv_data_x, verbose=False)
                print('Training has converged:', converged)
                counter+=1
                if counter==1:## if not converged in 1 trial, break and save the logits anyway
                  break
            print('Number of training trials:', counter)
            clusteringM.append(cluster_logits[0].squeeze())
        clustering.append(clusteringM)
                     
        pairwise_ari_results = {}
        for i, cluster in enumerate(clustering):
            pairwise_ari_results[M_list[i]] =  pairwise_ari(cluster)
            
        with open(os.path.join(opt['path'], f'pairwise_ari_dict.pkl'), 'wb') as f:
        	pickle.dump(pairwise_ari_results, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    return clustering

