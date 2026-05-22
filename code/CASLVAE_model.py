import os
import torch.distributions as D
import torch
import torch.nn.functional as F
import torch.nn as nn
from networks import Generator_CASL_twodecoders
from utils import weights_init, mmd, dotdict
from collections import OrderedDict


class CASLVAE(object):
    def __init__(self):
        self.opt = None

        ##### definition of all netwotks
        self.G_forward = None

        ##### definition of all optimizers
        # Optimizers
        self.optimizer_G = None
        
        ##### definition of all criterions
        self.criterionRecon = F.mse_loss # MSE loss
        self.mmd = mmd

        ##### define loss hyperparameters
        self.kl_z_lambda = None
        self.kl_scat_lambda = None
        self.kl_scont_lambda = None
        self.kl_sconty_lambda = None
        self.mmd_lambda = None
        self.consistency_lambda = None
        self.norm_salienty_lambda = None
        self.entropy_lambda = None

        ##### define scheduler variables
        self.lr_lambda = None
        self.lr_scheduler_G = None
        
        
    def create(self, opt):
        self.opt = dotdict({})
        for key in opt.keys():
            self.opt[key] = opt[key]

        ## define all netwotks
        # Generate control data and discriminate between real and generated
        self.G_forward = Generator_CASL_twodecoders(self.opt.nfeatures, self.opt.nLatent_z, self.opt.nLatent_s_cat,self.opt.nLatent_s_cont, self.opt.ncategories).to(self.opt.DEVICE)

        ## initiate networks
        #self.G_forward.apply(weights_init)


        ## define all optimizers
        self.optimizer_G = torch.optim.Adam(self.G_forward.parameters(),
                                        lr=self.opt.lr, betas=(self.opt.beta1, 0.999),weight_decay=2.5*1e-3)

        
        ## define all schedulers
        if self.opt.scheduler:
            self.lr_lambda = DecayLR(self.opt.epochs, 0, self.opt.decay_epochs).step
            self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=self.lr_lambda)

        ## define hyperparameters
        self.mmd_lambda=self.opt.mmd_lambda
        self.kl_z_lambda = self.opt.kl_z_lambda
        self.kl_scont_lambda=self.opt.kl_scont_lambda
        self.kl_scat_lambda=self.opt.kl_scat_lambda
        self.kl_sconty_lambda = self.opt.kl_sconty_lambda
        self.consistency_lambda=self.opt.consistency_lambda
        self.norm_salienty_lambda = self.opt.norm_salienty_lambda
        self.entropy_lambda = self.opt.entropy_lambda

    def train_instance(self, real_x, real_y, covariate_x=None, covariate_y=None, tau=1):
        ## Input to networks
        assert len(real_x.shape)==2 and real_x.shape[-1]==self.opt.nfeatures, 'real_x should be a 2D tensor with last dim equal to number of ROIs'
        assert len(real_y.shape)==2 and real_y.shape[-1]==self.opt.nfeatures, 'real_y should be a 2D tensor with last dim equal to number of ROIs'

        real_x = real_x.to(self.opt.DEVICE)
        real_y = real_y.to(self.opt.DEVICE)

        ## set networks to train mode
        self.G_forward.train()

        # get latents and reconstructions                
        output = self.G_forward(real_x, real_y, tau) 

        s_cat_x = output['s_cat_x']
        s_cat_y = output['s_cat_y']
        mu_s_x = output['mu_s_x']
        mu_sprior_x = output['mu_sprior_x']
        mu_s_y = output['mu_s_y']
        mu_sprior_y  = output['mu_sprior_y']
        z_x = output['z_x']
        z_y = output['z_y']
        mu_z_x = output['mu_z_x']
        logsigma_z_x = output['logsigma_z_x']
        mu_z_y = output['mu_z_y']
        logsigma_z_y = output['logsigma_z_y']
        y_recon_s = output['y_recon_s']
        
        gammas = torch.FloatTensor([10 ** x for x in range(-3, 3, 1)])     
        
        ###------------------------------------------------------------------------------------
        # (1) MMD between z_x, z_y
        ###-----------------------------------------------------------------------------------
        mmd_z_dist_loss =  self.mmd(z_x,z_y,gammas,self.opt.DEVICE) 
        #------------------------------------------------------------------------------------
        # (2) reconstruction loss
        #-------------------------------------------------------------------------------------
        recon_loss_x = self.criterionRecon(output['x_recon'], real_x, reduction='sum')/real_x.shape[0]
        recon_loss_y = self.criterionRecon(output['y_recon'], real_y, reduction='sum')/real_y.shape[0]
        #-----------------------------------------------------------------------------------------
        #3) KL divergence on  common latent z
        #-----------------------------------------------------------------------------------------
        kldivxy = (1 + 2*logsigma_z_x - torch.square(mu_z_x) - torch.square(logsigma_z_x.exp()))*0.5
        kldivxy += (1 + 2*logsigma_z_y - torch.square(mu_z_y) - torch.square(logsigma_z_y.exp()))*0.5
        kldivxy = -0.5*torch.sum(kldivxy)
        kldivxy /= real_x.shape[0]
        #-----------------------------------------------------------------------------------------
        #4) KL divergence on  salient discrete latent s_cat
        #-----------------------------------------------------------------------------------------
        q_dx = F.softmax(output['logits_x'], dim=-1) 
        log_q_dx = torch.log(q_dx + 1e-20)
        kldiv_s_cat = torch.sum(q_dx * (log_q_dx - torch.log(torch.tensor(1.0 / q_dx.size(-1)))), dim=-1)
        kldiv_s_cat = kldiv_s_cat.mean()
        #-----------------------------------------------------------------------------------------
        #4) KL divergence on  salient continuous latent s_cont
        #-----------------------------------------------------------------------------------------   
        # for x
        kldiv_s_cont = torch.square(mu_sprior_x-mu_s_x)
        kldiv_s_cont = 0.5*torch.sum(kldiv_s_cont)
        kldiv_s_cont /= real_x.shape[0]

        # for y, we want posterior and prior means to be 0
        kldiv_s_cont_y_penalty  = 0.5*torch.sum(torch.square(mu_s_y))/real_y.shape[0]
        kldiv_sprior_cont_y_penalty = 0.5*torch.sum(torch.square(mu_sprior_y))/real_y.shape[0]
        s_conty_penalty =  kldiv_s_cont_y_penalty + kldiv_sprior_cont_y_penalty 
        #-----------------------------------------------------------------------------------------
        #5) Penalty on salient reconstruction of y samples, L1norm(y_recon_s) =  0
        #------------------------------------------------------------------------------------------
        y_recon_s_loss =  torch.mean(torch.abs(y_recon_s))
        #-----------------------------------------------------------------------------------------
        #5) Entropy Loss, all y samples should be assigned to preset s_cat, and all x samples should not be assigned to preset s_cat
        #------------------------------------------------------------------------------------------
        entropy_loss = -torch.log(s_cat_y[:,0] + 1e-8).mean() -torch.log(1-s_cat_x[:,0] + 1e-8).mean()
        #--------------------------------------------------------------------------------------------
        #6) latent inverse consistency loss: ensures that target sample and its counterfactual paired background sample have same z
        #---------------------------------------------------------------------------------------------
        preset_scat = torch.zeros([real_x.shape[0],self.opt.ncategories],device=self.opt.DEVICE) # zdx is concatenation fo categorical and continuous zd latent, since for y continuous zd=0
        preset_scat[:,0] = 1 
        preset_scont = torch.zeros([real_x.shape[0],self.opt.nLatent_s_cont],device=self.opt.DEVICE) # zdx is concatenation fo categorical and continuous zd latent, since for y continuous zd=0
        counterfactual_y = self.G_forward.decode_common(z_x.detach()) + self.G_forward.decode_salient(preset_scat, preset_scont).detach()
        mu_z_counterfactual,_ = self.G_forward.encode_common(counterfactual_y) 
        consistency_loss = torch.mean(torch.abs(mu_z_x.detach()-mu_z_counterfactual)) # L1 norm 
                                                                   
        loss_G = recon_loss_x + recon_loss_y + self.kl_z_lambda*kldivxy + self.kl_scat_lambda*kldiv_s_cat + self.kl_scont_lambda*kldiv_s_cont + self.kl_sconty_lambda*s_conty_penalty + self.mmd_lambda*mmd_z_dist_loss + self.entropy_lambda*entropy_loss + self.consistency_lambda*consistency_loss
        ## update weights of model
        self.optimizer_G.zero_grad()
        loss_G.backward()
        self.optimizer_G.step()

        losses = OrderedDict([('Combined_loss', loss_G.item()), ('kldiv_z', kldivxy.item()), ('recon_loss_x', recon_loss_x.item()), ('recon_loss_y', recon_loss_y.item()), ('kldiv_s_cat', kldiv_s_cat.item()) , ('kldiv_sprior_cont_y', kldiv_sprior_cont_y_penalty.item()),
                 ('kldiv_s_cont', kldiv_s_cont.item()), ('mmd_z_loss', mmd_z_dist_loss.item()), ('entropy_loss', entropy_loss.item()), ('consistency_loss', consistency_loss.item()), ('y_recon_s_loss', y_recon_s_loss.item()), ('kldiv_s_cont_y', kldiv_s_cont_y_penalty.item())])
        return losses

    def set_preprocessing_params(self, data_params):
        self.train_mean = data_params['data_mean']
        self.train_std = data_params['data_std']
        self.covariatecorrection_params = {'cov_min': data_params['cov_min'],
                                            'cov_max': data_params['cov_max'],
                                            'cov_beta': data_params['cov_beta'],
                                            'covariate_correct': data_params['covariate_correct']
                                            }
        self.feature_names = data_params['feature_names']
        self.covariate_names =  data_params['covariate_names']
        self.tau=data_params['tau']#max


    def save_model(self, model, epoch, loss_list, opt, path='./', checkpointfilename='model_checkpoint.pt'):
        checkpoint = {
            'G_forward': model.G_forward.state_dict(),
            'optimizer_G': model.optimizer_G.state_dict(),
            # preprocessing stats
            'train_mean': model.train_mean,
            'train_std': model.train_std,
            # covariate correction parameters
            'covariatecorrection_params': model.covariatecorrection_params,
            # bookkeeping
            'loss_list': loss_list,
            'epoch': epoch,
            'feature_names': model.feature_names,
            'covariate_names': model.covariate_names,
            'opt':opt,
            'tau':loss_list['tau'][-1]
            }

        torch.save(checkpoint, os.path.join(path, checkpointfilename))


    def load_model(self, checkpoint_path, device="cuda"):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        checkpoint['opt']['DEVICE'] = device
        # Restore hyperparameters if saved
        self.create(checkpoint['opt'])

        # Load networks
        self.G_forward.load_state_dict(checkpoint['G_forward'])

        # Load optimizer
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])

        # Restore preprocessing stats
        self.train_mean = checkpoint.get('train_mean', None)
        self.train_std = checkpoint.get('train_std', None)
        self.covariatecorrection_params = checkpoint.get('covariatecorrection_params', None)

        # Restore metadata
        self.feature_names = checkpoint.get('feature_names', None)
        self.covariate_names = checkpoint.get('covariate_names', None)
        self.tau = checkpoint.get('tau',None)
        return checkpoint






