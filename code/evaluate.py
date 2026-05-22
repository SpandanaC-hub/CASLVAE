from scipy.linalg import sqrtm 
import scipy
import torch
import numpy as np


## calulate wasserstein distance assuming independent ROIs
def cal_w_distance(mean_1,mean_2,std_1,std_2):
    distance=np.sum(np. square(mean_1-mean_2) + np.square(std_1) + np.square(std_2) - (2*std_1*std_2))
    return  distance
  
## calulate wasserstein distance without assuming independent ROIs
def cal_w_distance_with_cov(mean_1,mean_2,std_1,std_2):
    distance=0
    distance=np.sum(np.square(mean_1-mean_2))+np.trace(std_1+std_2-2*sqrtm(sqrtm(std_2).dot(std_1).dot(sqrtm(std_2))))
    return  distance.real

## return mean wasserstein distance and wasserstein distance of each mapping direction
def cal_validate_distance(mean,std,eva_data,independent):
    mean_eva = np.mean(eva_data, axis=0)
    if independent:
        std_eva = np.std(eva_data, axis=0)
        distance = cal_w_distance(mean, mean_eva, std, std_eva)
    else:
        std_eva = np.cov(np.transpose(eva_data))
        distance = cal_w_distance_with_cov(mean,mean_eva, std, std_eva)
    return distance

def eval_w_distances(x,y,independent=True):
    # x, y (torch.tensor): x is first sample set of shape (N, D). y is second sample set of shape (M, D).
    # independent (bool): if True, assumes features are indepednent aka diagnosal covariance
    y=y.to('cpu').numpy()
    mean = np.mean(y, axis=0) # mean of each ROI in generated HC
    if independent:
        std = np.std(y, axis=0) 
    else:
        std = np.cov(np.transpose(y)) # covariance of each ROI wrt every other ROI in generated HC

    eva_data = x.to('cpu').numpy()
    w_distance = cal_validate_distance(mean,std,eva_data,independent)
    return w_distance

## return clustering memberships of datapoints and quantity of subjects assigned to each cluster
def label_change(logits,ncat):
    "logits (tensor): class probabilities, ncat (int): number of classes"
    predicted_class = []
    predicted_label = torch.argmax(logits, dim=-1).to('cpu').numpy()
    for i in range(ncat):
        predicted_class.append(sum(predicted_label==i))
    return predicted_label, predicted_class


def sample_cavae(model, num_samples, type='gumbel_pairs'):
    model.G_forward.eval()
    with torch.no_grad():
        device = model.opt.DEVICE
        if type=='gumbel_pairs':
            # sample common latnet z
            z_sample = torch.randn(num_samples,model.opt.nLatent_z, device=device)
            # sample salient category
            scat_sample = model.G_forward.gumbel_sample(num_samples=num_samples,device=device)
            # sample intra category salient latnet scont
            mucontprior = model.G_forward.prior_s_cont(scat_sample)
            scont_sample = mucontprior+torch.randn(num_samples,model.opt.nLatent_s_cont,device=device)
            # define preset saleint
            scat_empty = torch.zeros([num_samples, model.opt.nLatent_s_cat, model.opt.ncategories], device=device)
            scat_empty[:,:,0]=1
            scat_empty = scat_empty.view(-1, model.opt.nLatent_s_cat*model.opt.ncategories)
            scont_empty = torch.zeros(num_samples,model.opt.nLatent_s_cont,device=device)
            # generate
            synth_x = model.G_forward.decode_common(z_sample) + model.G_forward.decode_salient(scat_sample, scont_sample)
            synth_y = model.G_forward.decode_common(z_sample) + model.G_forward.decode_salient(scat_empty, scont_empty)
        else: # gumbel unpaired
            # sample common latent z
            z_sample_1 = torch.randn(num_samples,model.opt.nLatent_z, device=device)
            z_sample_2 = torch.randn(num_samples,model.opt.nLatent_z, device=device)
            # sample salient category=
            scat_sample = model.G_forward.gumbel_sample(num_samples=num_samples,device=device)
            # sample intra category salient latnet scont
            mucontprior = model.G_forward.prior_s_cont(scat_sample)
            scont_sample = mucontprior+torch.randn(num_samples,model.opt.nLatent_s_cont,device=device)
            # define preset saleint
            scat_empty = torch.zeros([num_samples, model.opt.nLatent_s_cat, model.opt.ncategories], device=device)
            scat_empty[:,:,0]=1
            scat_empty = scat_empty.view(-1, model.opt.nLatent_s_cat*model.opt.ncategories)
            scont_empty = torch.zeros(num_samples,model.opt.nLatent_s_cont, device=device)
            # generate
            synth_x = model.G_forward.decode_common(z_sample_1) + model.G_forward.decode_salient(scat_sample, scont_sample)
            synth_y = model.G_forward.decode_common(z_sample_2) + model.G_forward.decode_salient(scat_empty, scont_empty)
        synth_x = synth_x.to('cpu').numpy()
        synth_y = synth_y.to('cpu').numpy()
        scat_sample = scat_sample.to('cpu').numpy()
    return synth_x, synth_y, scat_sample



def infer_latentembedding(model, input_data, input_covariates=None):
    """ infer latent representations
    model: loaded trained model
    input_data: numpy array of size N x D
    input_covariates: numpy_array of size N x P
    """
    # ----------------------------- 
    # 1. Check preprocessing params  and preprocess 
    #  -----------------------------
    if model.covariatecorrection_params['covariate_correct']:
        if input_covariates is None:
            raise ValueError("No covariates found, model requires covariate correction.")
        else: # covariate correct and normalize
            inp_cov = (input_covariates-model.covariatecorrection_params['cov_min'])/(model.covariatecorrection_params['cov_max']-model.covariatecorrection_params['cov_min'])
            data_corr = input_data - np.dot(inp_cov, model.covariatecorrection_params['cov_beta'])
            data_norm = (data_corr - model.train_mean)/model.train_std
            data_norm = torch.from_numpy(data_norm).to(model.opt.DEVICE).type(torch.float32)
    else: 
        data_norm = (input_data - model.train_mean)/model.train_std
        data_norm = torch.from_numpy(data_norm).to(model.opt.DEVICE).type(torch.float32)

    # ----------------------------- 
    #  2. Encode latents 
    # -----------------------------
    model.G_forward.eval()
    with torch.no_grad():
        device = model.opt.DEVICE
        mu_z, logsigma_z = model.G_forward.encode_common(data_norm)
        logits, proj_input = model.G_forward.encode_salient_cat(data_norm)
        idx = logits.argmax(dim=-1, keepdim=True)
        scat = torch.zeros_like(logits).scatter_(-1, idx, 1.0)
        mu_scont = model.G_forward.enc_s_cont(torch.cat([proj_input, scat],dim=-1)) 

    return {'common_latent_mean': mu_z.to('cpu').numpy(), 
            'common_latent_logsigma': logsigma_z.to('cpu').numpy(), 
            'salient_logits': logits.to('cpu').numpy(), 
            'salient_encoder_projection': proj_input.to('cpu').numpy(), 
            'salient_cluster': scat.to('cpu').numpy(),
            'salient_intracluster_mean': mu_scont.to('cpu').numpy(),
            'preprocessed_inputdata': data_norm.to('cpu').numpy()}

def infer_reconstruction(model,input_data):
    """ infer input reconstruction its counterfactual y 
    model: loaded trained model
    input_data: numpy array of size N x D after pre-processing
    """
    # ----------------------------- 
    #  Infer latents and reconstruct
    # -----------------------------
    model.G_forward.eval()
    with torch.no_grad():
        device = model.opt.DEVICE
        input_data = torch.from_numpy(input_data).to(device).type(torch.float32)
        mu_z, logsigma_z = model.G_forward.encode_common(input_data)
        logits, proj_input = model.G_forward.encode_salient_cat(input_data)
        idx = logits.argmax(dim=-1, keepdim=True)
        scat = torch.zeros_like(logits).scatter_(-1, idx, 1.0)
        mu_scont = model.G_forward.enc_s_cont(torch.cat([proj_input, scat],dim=-1)) 
        inp_recon = model.G_forward.decode_common(mu_z) + model.G_forward.decode_salient(scat, mu_scont) # decoding posterior means
        scat_preset = torch.zeros_like(scat)
        scat_preset[:,0]=1
        scont_preset = torch.zeros_like(mu_scont)
        inp_counterfactual_y = model.G_forward.decode_common(mu_z) + model.G_forward.decode_salient(scat_preset, scont_preset) # decoding counterfactual y 
        s_cluster_index =  scipy.special.softmax(logits.detach().to('cpu').numpy().squeeze(),axis=-1)
        
        return {'inp_reconstruction': inp_recon.to('cpu').numpy(), 
        'inp_counterfactual_y': inp_counterfactual_y.to('cpu').numpy(), 
        'salient_cluster_index': s_cluster_index
        }


