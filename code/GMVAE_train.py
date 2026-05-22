
import os, time, numpy as np, math
from utils import format_time,  weights_init,  dotdict
from evaluate import eval_w_distances , label_change
import torch
import torch.nn.functional as F
import torch.distributions as D
import torch.nn as nn
from networks import GMVAE_network
from collections import OrderedDict

import matplotlib.pyplot as plt

class GMVAE(object):
    def __init__(self):
        self.opt = None

        ##### definition of all netwotks
        self.G_forward = None

        ##### definition of all optimizers
        # Optimizers
        self.optimizer_G = None

        ##### definition of all criterions
        self.criterionRecon = F.mse_loss # MSE loss

        ##### define loss hyperparameters
        self.kl_scat_lambda = None
        self.kl_scont_lambda = None

        ##### define scheduler variables
        self.lr_lambda = None
        self.lr_scheduler_G = None
        
        
    def create(self, opt):
        self.opt = dotdict({})
        for key in opt.keys():
            self.opt[key] = opt[key]

        ## define all netwotks
        # Generate control data and discriminate between real and generated
        self.G_forward = GMVAE_network(self.opt.nfeatures, self.opt.nLatent_s_cat, self.opt.nLatent_s_cont, self.opt.ncategories).to(self.opt.DEVICE)

        ## initiate networks
        self.G_forward.apply(weights_init)


        ## define all optimizers
        self.optimizer_G = torch.optim.Adam(self.G_forward.parameters(),
                                        lr=self.opt.lr, betas=(self.opt.beta1, 0.999),weight_decay=2.5*1e-3)

        
        ## define all schedulers
        if self.opt.scheduler:
            self.lr_lambda = DecayLR(self.opt.epochs, 0, self.opt.decay_epochs).step
            self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=self.lr_lambda)

        ## define hyperparameters
        self.kl_scont_lambda=self.opt.kl_scont_lambda
        self.kl_scat_lambda=self.opt.kl_scat_lambda

    def log_normal(self, x, mu, var, eps=1e-6):
        var = torch.clamp(var, eps, 1e6)

        return -0.5 * torch.sum(math.log(2 * math.pi) + torch.log(var) + torch.square(x - mu) / var, dim=-1)

    def loss_per_category(self, real_input, recon_input, samples_scont, mu_scont, logsigma_scont, mu_scontprior, logsigma_scontprior):
        logsigma_scont = torch.clamp(logsigma_scont, -10, 10) 
        logsigma_scontprior = torch.clamp(logsigma_scontprior, -10, 10)
        #------------------------------------------------------------------------------------
        # (1) reconstruction loss
        #-------------------------------------------------------------------------------------
        recon_loss_x = self.criterionRecon(recon_input, real_input, reduction='none').sum(axis=-1)

        #-----------------------------------------------------------------------------------------
        #2) KL divergence on  salient continuous latent s_cont
        #-----------------------------------------------------------------------------------------   
        # get KL terms (log(q(scont/scat,x))-log(p(scont/scat))) which will later be weighted wrt the class probabilities
        # compute variance safely 
        var_scont = torch.exp(2 * logsigma_scont) 
        var_scontprior = torch.exp(2 * logsigma_scontprior)
        kldiv_s_cont = self.log_normal(samples_scont,mu_scont,var_scont) - self.log_normal(samples_scont,mu_scontprior,var_scontprior)

        return recon_loss_x,  kldiv_s_cont , np.log(1/self.opt.ncategories)

    def train_instance(self, real_x, real_y=None, label = None, covariate_x=None, covariate_y=None):
        ## Input to networks
        assert len(real_x.shape)==2 and real_x.shape[-1]==self.opt.nfeatures, 'real_x should be a 2D tensor with last dim equal to number of ROIs'

        real_x = real_x.to(self.opt.DEVICE)

        ## set networks to train mode
        self.G_forward.train()

        # get latents and reconstructions                
        output , _= self.G_forward(real_x) 

        s_cat_x = output['s_cat_x']
        mu_s_x = output['mu_s_x']
        mu_sprior_x = output['mu_sprior_x']
        logsigma_s_x = output['logsigma_s_x']
        logsigma_sprior_x  = output['logsigma_sprior_x']
        s_cont_x = output['s_cont_x']
        logits_x = output['logits_x'].view(-1, self.opt.nLatent_s_cat*self.opt.ncategories) 
        x_recon = output['x_recon']
        #-----------------------------------------------------------------------------------------
        # KL divergence on discrete latent term log q(scat/x)
        #-----------------------------------------------------------------------------------------
        kldiv_s_cat_term = torch.sum(s_cat_x * torch.nn.LogSoftmax(1)(logits_x),1)

        losses_perclass = [] 
        for i in range(self.opt.ncategories):
            recon_loss, kl_loss, kl_cat_term = self.loss_per_category(real_x, x_recon[i],s_cont_x[i],mu_s_x[i],logsigma_s_x[i], mu_sprior_x[i], logsigma_sprior_x[i])
            losses_perclass.append(recon_loss+ kl_loss- kl_cat_term)
        loss = torch.stack([kldiv_s_cat_term] + [s_cat_x[:, i] * losses_perclass[i] for i in range(self.opt.ncategories)]).sum(0)
        loss_G = loss.sum()
        
        ## update weights of model
        self.optimizer_G.zero_grad()
        loss_G.backward()
        self.optimizer_G.step()

        losses = OrderedDict([('Combined_loss', loss.mean().item()), ('NLL', -kldiv_s_cat_term.mean().item()), ('recon_loss', recon_loss.sum().item()), ('kl_loss', kl_loss.sum().item())])
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


    def save_model(self, model, epoch, loss_list, opt, path='./', checkpointfilename='model_checkpoint_gmvae.pt'):
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
            'opt':opt
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
        return checkpoint





def train_GMVAE(model, opt, train_loader, train_data_ref, train_data_pt, cv_data_ref, cv_data_pt):
            # initialize variables to store losses
            clusteringM=[]
            G_combined_loss = []
            reconx_loss_eva= []
            w_list_hc=[] # WD between real HC and fake HC
            w_list_eva_hc=[]
            w_list_pt=[]# WD between real PT and fake PT
            w_list_eva_pt=[]
            w_list_pair=[] # WD between real HC and real PT
            w_list_eva_pair=[] # WD between fake HC and fake PT
            entropy_loss=[]
            n_batches=len(train_loader)

            reduce = 'mean'
            predicted_label_past = np.zeros(cv_data_pt.shape[0])
            cluster_stability = [0,0,0,0] # lets monitor over the last 4 evaluations
            max_epochs = opt['epochs']
            min_epochs_for_earlystopping = max_epochs/2
            patience = 20
            tolerance = 1e-1
            for epoch in range(max_epochs):
                G_combined_batch_loss = 0
                entropy_batch=0
                print("")
                print('======== Epoch {:} / {:} ========'.format(epoch+1, opt['epochs']))
                print('Training...')
                t0 = time.time()
                for i,(ref_batch, pt_batch) in enumerate(train_loader):
                    ref_batch = ref_batch.to(opt['DEVICE']).type(torch.float32)
                    pt_batch = pt_batch.to(opt['DEVICE']).type(torch.float32)
                    #train
                    losses = model.train_instance(pt_batch, None)
                    G_combined_batch_loss += losses['Combined_loss']
                    entropy_batch += losses['NLL']
                    # Progress update every 40 batches.
                #if i % 40 == 0 and not i == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                # Report progress.
                #print('  Batch {:>5,}  of  {:>5,}. \nLosses {}'.format(i, n_batches, losses))

                G_combined_loss.append(G_combined_batch_loss/n_batches) # combined loss
                entropy_loss.append(entropy_batch/n_batches) # combined loss

                print("")
                print("  Average loss: {0:.2f}".format(G_combined_loss[-1]))
                #print("  Average recon x loss: {0:.2f}".format(reconx_loss[-1]))
                #print("  Average recon y loss: {0:.2f}".format(recony_loss[-1]))
                print("  Training epoch took: {:}".format(format_time(time.time() - t0)))

                # model eval
                model.G_forward.eval()
                with torch.no_grad():
                    _, output = model.G_forward(train_data_pt)
                    _, output_eva = model.G_forward(cv_data_pt)
                    reconx_loss_eva.append(float(F.mse_loss(output_eva['x_recon'], cv_data_pt, reduction='sum')/cv_data_pt.shape[0]))

                    scat_synth = F.one_hot(torch.randint(0,opt['ncategories'],(400,1),device=opt['DEVICE']) ).squeeze().type(torch.float32)
                    #mu_sprior_synth, logsigma_sprior_synth = model.G_forward.prior_cont(scat_synth).chunk(2,-1)
                    mu_sprior_synth = model.G_forward.prior_cont(scat_synth)#.chunk(2,-1)#torch.zeros(400, opt['nLatent_s_cont'],device=opt['DEVICE'])#
                    logsigma_sprior_synth = torch.zeros_like(mu_sprior_synth)
                    scont_synth = model.G_forward.gaussian_sample(mu_sprior_synth, logsigma_sprior_synth)
                    
                    synth_pt = model.G_forward.dec(scont_synth) 

                    w_list_pt.append(eval_w_distances(train_data_pt, output['x_recon'], independent=False))
                    w_list_eva_pt.append(eval_w_distances(cv_data_pt, output_eva['x_recon'],  independent=False))
                    w_list_pair.append(eval_w_distances(train_data_ref, train_data_pt, independent=False))
                    w_list_eva_pair.append(eval_w_distances(train_data_ref, synth_pt, independent=False))
                    if epoch%5==0:
                        predicted_label, predicted_class = label_change(output_eva['logits_x'],opt['ncategories'])
                        #print("")
                        print("class memebership ", predicted_class)
                        total_label_change = np.count_nonzero(np.absolute(np.array(predicted_label)-np.array(predicted_label_past))) 
                        print("Total label change ", total_label_change)
                        predicted_label_past = predicted_label
                        cluster_stability.append(total_label_change)
                        cluster_stability.pop(0)
                        if epoch>(min_epochs_for_earlystopping):
                            if (max(cluster_stability)< max(int(cv_data_pt.shape[0]*0.01),10)) and (reconx_loss_eva[-1] > min(reconx_loss_eva[-patience:])+tolerance):
                                loss_list = {'combined_loss': G_combined_loss, 'recon_x': [ reconx_loss_eva], 'kl_cat_term':[entropy_loss], 'w_list_hc':[w_list_hc, w_list_eva_hc], 'w_list_pt':[w_list_pt, w_list_eva_pt],'w_list_pair':[w_list_pair, w_list_eva_pair]}
                                model.save_model(model, epoch, loss_list, opt, path=opt['path'])
                                # model eval
                                logits_eva=output_eva['logits_x'].view(-1,1,opt['ncategories']).to('cpu').numpy() 
                                # for clustering stability we want to evaluate without gumbel sampling
                                clusteringM.append(logits_eva)
                                return model, clusteringM, loss_list, True
                
                if epoch%50==0:
                    for i in range(8):
                        ax=plt.subplot(2,4,i+1)
                        ax.scatter(cv_data_pt[:,0].to('cpu').numpy(),cv_data_pt[:,i].to('cpu').numpy(), alpha=0.2)
                        ax.scatter(cv_data_ref[:,0].to('cpu').numpy(),cv_data_ref[:,i].to('cpu').numpy(), alpha=0.2)
                        ax.scatter(output_eva['x_recon'][:,0].to('cpu').numpy(),output_eva['x_recon'][:,i].to('cpu').numpy(), alpha=0.2)
                    plt.legend(['real PT','real HC', 'fake PT'])
                    plt.savefig(os.path.join(opt['path'],f'GMVAE_quality_epoch{epoch}.jpg'))
                    plt.close()
            
            loss_list = {'combined_loss': G_combined_loss, 'recon_x': [ reconx_loss_eva], 'kl_cat_term':[entropy_loss],   'w_list_hc':[w_list_hc, w_list_eva_hc], 'w_list_pt':[w_list_pt, w_list_eva_pt],'w_list_pair':[w_list_pair, w_list_eva_pair]}
            
            model.save_model(model, epoch, loss_list, opt, path=opt['path'])
            # save cluster memebership
            logits_eva=output_eva['logits_x'].view(-1,1,opt['ncategories']).to('cpu').numpy() 
            clusteringM.append(logits_eva)
            return model, clusteringM, loss_list, False


