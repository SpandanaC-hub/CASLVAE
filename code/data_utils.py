from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import sklearn
import numpy as np
from sklearn.linear_model import LinearRegression

class PreprocessingStats: # preprocessing params
    def __init__(self, data, labels, covariates=None, covariate_correct=False):
        self.data = data.copy() # numpy array size N x D
        self.label = labels.copy() # numpy array size N 
        self.covariates = covariates # numpy array size N X P
        self.covariate_correct = covariate_correct

    def stats(self):
        background_mask = (self.label=='Background')
        if background_mask.sum() == 0: 
            raise ValueError("data_utils.PreprocessingStats: No 'Background' samples found in labels.")
        background_data = self.data[background_mask] 
        if self.covariate_correct and (self.covariates is not None): # get covariate correction params and data normalization params
            background_covariates = self.covariates[background_mask].copy()
            self.cov_min = np.min(background_covariates,axis=0)
            self.cov_max = np.max(background_covariates,axis=0)
            if np.any(self.cov_max == self.cov_min): 
                raise ValueError("data_utils.PreprocessingStats: Covariate has zero variance; remove it from covariate_cols.")
            background_covariates_standardized  = (background_covariates-self.cov_min)/(self.cov_max-self.cov_min)
            self.cov_beta  = np.transpose(LinearRegression().fit(background_covariates_standardized, background_data).coef_)
            covariate_effects = np.dot(background_covariates_standardized, self.cov_beta)
            background_data_corr = background_data - covariate_effects
            self.data_mean = np.mean(background_data_corr, axis=0)
            self.data_std = np.std(background_data_corr, axis=0)
        else: # get data normalziation params
            self.cov_min = None
            self.cov_max = None
            self.cov_beta = None
            self.data_mean = np.mean(background_data, axis=0)
            self.data_std = np.std(background_data, axis=0)

class PreprocessingTransform:
    def __init__(self, data, stats, covariates=None, covariate_correct=False):
        self.data = data.copy() # numpy array size N x D
        self.covariates = covariates # numpy array size N X P
        self.covariate_correct = covariate_correct
        self.preprocessingstats =  stats

    def covariate_correction(self):
        self.cov = (self.covariates.copy()-self.preprocessingstats.cov_min)/(self.preprocessingstats.cov_max-self.preprocessingstats.cov_min)
        self.cov_effects = np.dot(self.cov, self.preprocessingstats.cov_beta)
        self.data_corr =  self.data - self.cov_effects

    def transform(self):
        if self.covariate_correct:
            self.covariate_correction()
            return (self.data_corr-self.preprocessingstats.data_mean)/(self.preprocessingstats.data_std+1e-6)
        else:
            return (self.data-self.preprocessingstats.data_mean)/(self.preprocessingstats.data_std+1e-6)


def train_test_split_unpaired(df, feature_cols, label_col, covariate_cols=[], test_size=0.3, stratify=False, covariate_correct=False,random_state=None, transform=True):
    """
    df: dataframe with data
    feature_cols: list of columns names of features
    feature_cols: columns name of label
    covariate_cols: list with column names of covarites
    test_size: ratio of split size
    stratify: if True, stratifies based on label
    covarite_correct: if True, covariate corrects the features 
    """
    # 1. Split indices
    if stratify:
        train_idx, test_idx = sklearn.model_selection.train_test_split(
            df.index, test_size=test_size, random_state=random_state, stratify=df[label_col])
    else:
        train_idx, test_idx = sklearn.model_selection.train_test_split(
            df.index, test_size=test_size, random_state=random_state)

    # 2. Extract arrays
    train_data = df.loc[train_idx, feature_cols].to_numpy()
    train_label = df.loc[train_idx, label_col].to_numpy()
    train_covariate = df.loc[train_idx, covariate_cols].to_numpy()

    test_data = df.loc[test_idx, feature_cols].to_numpy()
    test_label = df.loc[test_idx, label_col].to_numpy()
    test_covariate = df.loc[test_idx, covariate_cols].to_numpy()
    
    if covariate_correct and (train_covariate.shape[1]==0):
        raise ValueError("data_utils: covariate_correct is True but empty covariate_cols.")

    preprocess_fn = PreprocessingStats(train_data, train_label, train_covariate, covariate_correct=covariate_correct)
    preprocess_fn.stats()

    if transform:
      train_data = PreprocessingTransform(train_data, preprocess_fn,  train_covariate, covariate_correct).transform()
      test_data = PreprocessingTransform(test_data, preprocess_fn,  test_covariate, covariate_correct).transform()
    else:
      pass
    return train_data, train_label, train_covariate, test_data, test_label, test_covariate, preprocess_fn

class CreateDataset(Dataset):
    def __init__(self, data, labels):
        self.bg_mask = (labels=='Background')
        self.data_bg = data[self.bg_mask,:]      # e.g. a tensor or numpy array
        self.data_tgt = data[~self.bg_mask,:]
        self.bg_len = self.data_bg.shape[0]
        self.tgt_len = self.data_tgt.shape[0]
        self.bg_labels = labels[self.bg_mask]
        self.tgt_labels = labels[~self.bg_mask]
        if self.bg_len == 0 or self.tgt_len == 0: 
            raise ValueError("Dataset must contain both Background and Target samples.")

    def __len__(self):
        return max(self.bg_len, self.tgt_len)

    def __getitem__(self, idx):
        # REQUIRED: must return a single sample 
        bg = self.data_bg[idx%self.bg_len]
        tgt = self.data_tgt[idx%self.tgt_len]
        return bg, tgt
        
        
# get preprocessing params
def get_covariatecorrection_params(data, covariates, label):
    background_mask=(label=='Background')
    background_data = data[background_mask].copy()

    background_covariates = covariates[background_mask].copy()
    
    cov_min = np.min(background_covariates,axis=0)
    cov_max = np.max(background_covariates,axis=0)
    if np.any(cov_max == cov_min): 
        raise ValueError("data_utils.covariate correction: Covariate has zero variance; remove it from covariate_cols.")
    background_covariates_standardized  = (background_covariates - cov_min)/(cov_max - cov_min)
    cov_beta  = np.transpose(LinearRegression().fit(background_covariates_standardized, background_data).coef_)
    return cov_min, cov_max, cov_beta

def get_normalization_params(data, label):
    background_mask = (label=='Background')
    background_data = data[background_mask].copy()
    mean = np.mean(background_data, axis=0)
    std = np.std(background_data, axis=0)
    return mean, std
    
# covariate correction
def apply_covariate_correction(data, covariates, covariatecorrection_params):
    # data,covartiates is numpy array, covariatecorrection_params is dict with parameters
    x_cov = (covariates-covariatecorrection_params['cov_min'])/(covariatecorrection_params['cov_max']-covariatecorrection_params['cov_min'])
    cov_effects = np.dot(x_cov,covariatecorrection_params['cov_beta'])
    data_corr = data-cov_effects
    return data_corr


# normalize data
def apply_normalization(data, mean=0, std=1):
    # data is numpy array
    data_norm = (data-mean)/std
    return data_norm