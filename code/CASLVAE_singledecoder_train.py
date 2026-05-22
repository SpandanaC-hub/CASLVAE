import os, time, numpy as np, math
from utils import format_time,  weights_init,  dotdict, sample_gumbel_softmax, mmd
from evaluate import eval_w_distances , label_change
import torch
import torch.nn.functional as F
import torch.distributions as D
import torch.nn as nn
from networks import Generator_CASL_singledecoder
from collections import OrderedDict

import matplotlib.pyplot as plt



class CASLVAE_singledecoder(object):
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
        self.entropy_lambda = None
        self.norm_salienty_lambda = None

        ##### define scheduler variables
        self.lr_lambda = None
        self.lr_scheduler_G = None
        
        
    def create(self, opt):
        self.opt = dotdict({})
        for key in opt.keys():
            self.opt[key] = opt[key]

        ## define all netwotks
        # Generate control data and discriminate between real and generated
        self.G_forward = Generator_CASL_singledecoder(self.opt.nfeatures, self.opt.nLatent_z, self.opt.nLatent_s_cat,self.opt.nLatent_s_cont, self.opt.ncategories).to(self.opt.DEVICE)

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
        self.mmd_lambda=self.opt.mmd_lambda
        self.kl_z_lambda = self.opt.kl_z_lambda
        self.kl_scont_lambda=self.opt.kl_scont_lambda
        self.kl_scat_lambda=self.opt.kl_scat_lambda
        self.kl_sconty_lambda = self.opt.kl_sconty_lambda
        self.consistency_lambda=self.opt.consistency_lambda
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
        #5) Entropy Loss, all y samples should be assigned to preset s_cat, and all x samples should not be assigned to preset s_cat
        #------------------------------------------------------------------------------------------
        entropy_loss = -torch.log(s_cat_y[:,0] + 1e-8).mean() -torch.log(1-s_cat_x[:,0] + 1e-8).mean()
        #--------------------------------------------------------------------------------------------
        #6) latent inverse consistency loss: ensures that target sample and its counterfactual paired background sample have same z
        #---------------------------------------------------------------------------------------------
        preset_scat = torch.zeros([real_x.shape[0],self.opt.ncategories],device=self.opt.DEVICE) # zdx is concatenation fo categorical and continuous zd latent, since for y continuous zd=0
        preset_scat[:,0] = 1 
        preset_scont = torch.zeros([real_x.shape[0],self.opt.nLatent_s_cont],device=self.opt.DEVICE) # zdx is concatenation fo categorical and continuous zd latent, since for y continuous zd=0
        counterfactual_y = self.G_forward.decoder(z_x.detach(), preset_scat, preset_scont) 
        mu_z_counterfactual,_ = self.G_forward.encode_common(counterfactual_y) 
        consistency_loss = torch.mean(torch.abs(mu_z_x.detach()-mu_z_counterfactual)) # L1 norm 
                                                                   
        loss_G = recon_loss_x + recon_loss_y + self.kl_z_lambda*kldivxy + self.kl_scat_lambda*kldiv_s_cat + self.kl_scont_lambda*kldiv_s_cont + self.kl_sconty_lambda*s_conty_penalty + self.entropy_lambda*entropy_loss + self.consistency_lambda*consistency_loss + self.mmd_lambda*mmd_z_dist_loss 
        
        ## update weights of model
        self.optimizer_G.zero_grad()
        loss_G.backward()
        self.optimizer_G.step()

        losses = OrderedDict([('Combined_loss', loss_G.item()), ('kldiv_z', kldivxy.item()), ('recon_loss_x', recon_loss_x.item()), ('recon_loss_y', recon_loss_y.item()), ('kldiv_s_cat', kldiv_s_cat.item()) , ('kldiv_sprior_cont_y', kldiv_sprior_cont_y_penalty.item()),
                 ('kldiv_s_cont', kldiv_s_cont.item()), ('mmd_z_loss', mmd_z_dist_loss.item()), ('entropy_loss', entropy_loss.item()), ('consistency_loss', consistency_loss.item()), ('kldiv_s_cont_y', kldiv_s_cont_y_penalty.item())])
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
        self.tau=1 # set max temeprature


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





def train_model_singledecoder(model, opt, train_loader, train_data_ref, train_data_pt, cv_data_ref, cv_data_pt):
            # initialize variables to store losses
            clusteringM=[]
            kl_z_loss = []
            mmd_z_loss = []
            kl_scat_loss = []
            kl_scont_loss = []
            kl_scont_loss_y = []
            kl_scontprior_loss_y = []
            reconx_loss = []
            recony_loss = []
            G_combined_loss = []
            entropy_list =[]
            reconx_loss_eva = []
            recony_loss_eva = []

            w_list_hc=[] # WD between real HC and fake HC
            w_list_eva_hc=[]
            w_list_pt=[]# WD between real PT and fake PT
            w_list_eva_pt=[]
            w_list_pair=[] # WD between real HC and real PT
            w_list_eva_pair=[] # WD between fake HC and fake PT

            consistency_list = []

            n_batches=len(train_loader)
            # define temperature tau
            tau_min=0.3
            tau0=1
            max_epochs=opt['epochs']
            min_epochs_for_earlystopping = int(max_epochs/2)
            r = -np.log(tau_min/tau0)/(max_epochs*n_batches)

            reduce = 'mean'
            step=0
            tau_list = []
            predicted_label_past = np.zeros(cv_data_pt.shape[0])
            cluster_stability = [0,0,0,0] # lets monitor over the last 4 evaluations
            patience = 20
            tolerance = 1e-1
            for epoch in range(max_epochs):
                kl_z_batch = 0 
                recon_x_batch = 0
                recon_y_batch = 0
                kl_scat_batch = 0
                G_combined_batch_loss = 0
                kl_scont_batch = 0
                kl_scont_y_batch = 0
                kl_scontprior_y_batch = 0
                mmd_z_dist_batch = 0               
                entropy_batch = 0
                norm_reconsalient_y_batch = 0
                consistency_batch = 0 
                print("")
                print('======== Epoch {:} / {:} ========'.format(epoch+1, opt['epochs']))
                print('Training...')
                t0 = time.time()
                for i,(ref_batch, pt_batch) in enumerate(train_loader):
                    ref_batch = ref_batch.to(opt['DEVICE']).type(torch.float32)
                    pt_batch = pt_batch.to(opt['DEVICE']).type(torch.float32)
                    if step%100==0:
                        tau = max(tau_min, tau0 * np.exp(-r * step))
                    step+=1
                    tau_list.append(tau)
                    #train
                    losses = model.train_instance(pt_batch, ref_batch,tau=tau)
                    kl_z_batch += losses['kldiv_z'] # kl on z
                    G_combined_batch_loss += losses['Combined_loss']
                    recon_y_batch += losses['recon_loss_y']
                    recon_x_batch += losses['recon_loss_x']
                    kl_scat_batch += losses['kldiv_s_cat'] # kl on disease categorical latent
                    kl_scont_batch += losses['kldiv_s_cont'] # kl on disease continuous latent
                    mmd_z_dist_batch += losses['mmd_z_loss'] # MMD on z
                    entropy_batch += losses['entropy_loss']  # entropy on s cat 1
                    kl_scont_y_batch += losses['kldiv_s_cont_y'] # L2 norm on scont for y
                    kl_scontprior_y_batch += losses['kldiv_sprior_cont_y'] # L2 norm on scont for y
                    consistency_batch += losses['consistency_loss']
                    # Progress update every 40 batches.
                #if i % 40 == 0 and not i == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                # Report progress.
                #print('  Batch {:>5,}  of  {:>5,}. \nLosses {}'.format(i, n_batches, losses))

                kl_z_loss.append(kl_z_batch/n_batches) # G adv loss
                kl_scat_loss.append(kl_scat_batch/n_batches) # G adv loss
                kl_scont_loss.append(kl_scont_batch/n_batches) # d latent 
                mmd_z_loss.append(mmd_z_dist_batch/n_batches) # d latent 
                reconx_loss.append(recon_x_batch /n_batches) # recon c loss
                G_combined_loss.append(G_combined_batch_loss/n_batches) # combined loss
                recony_loss.append(recon_y_batch/n_batches) # recon y
                entropy_list.append(entropy_batch/n_batches)
                kl_scont_loss_y.append(kl_scont_y_batch/n_batches) # d latent 
                kl_scontprior_loss_y.append(kl_scontprior_y_batch/n_batches) # d latent 
                consistency_list.append(consistency_batch/n_batches)
                print("")
                print("  Average loss: {0:.2f}".format(G_combined_loss[-1]))
                #print("  Average recon x loss: {0:.2f}".format(reconx_loss[-1]))
                #print("  Average recon y loss: {0:.2f}".format(recony_loss[-1]))
                print("  Training epoch took: {:}".format(format_time(time.time() - t0)))

                # model eval
                model.G_forward.eval()
                with torch.no_grad():
                    output = model.G_forward(train_data_pt, train_data_ref,tau=tau)
                    output_eva = model.G_forward(cv_data_pt, cv_data_ref,tau=tau)


                    reconx_loss_eva.append(float(F.mse_loss(output_eva['x_recon'], cv_data_pt, reduction='sum')/cv_data_pt.shape[0]))
                    recony_loss_eva.append(float(F.mse_loss(output_eva['y_recon'], cv_data_ref, reduction='sum')/cv_data_ref.shape[0]))

                    z_synth = torch.randn(800,opt['nLatent_z'], device='cuda')
                    scont_synth = torch.randn(400,opt['nLatent_s_cont'], device='cuda')
                    
                    scat_preset_synth = torch.zeros(400,opt['ncategories'], device='cuda')
                    scat_preset_synth[:,0]=1
                    scont_preset_synth =   torch.zeros_like(scont_synth) 

                    scat_synth = torch.zeros(400,opt['ncategories'], device='cuda')
                    scat_synth[:,1:] = sample_gumbel_softmax(torch.zeros(400,opt['ncategories']-1) ,hard=True)
                    mu_sprior_synth = model.G_forward.prior_s_cont(scat_synth)
                    
                    synth_hc = model.G_forward.decoder(z_synth[:400], scat_preset_synth, scont_preset_synth)
                    synth_pt = model.G_forward.decoder(z_synth[400:], scat_synth, scont_synth+mu_sprior_synth)


                    w_list_hc.append(eval_w_distances(train_data_ref, output['y_recon'], independent=False))
                    w_list_eva_hc.append(eval_w_distances(cv_data_ref, output_eva['y_recon'],  independent=False))
                    w_list_pt.append(eval_w_distances(train_data_pt, output['x_recon'], independent=False))
                    w_list_eva_pt.append(eval_w_distances(cv_data_pt, output_eva['x_recon'],  independent=False))
                    w_list_pair.append(eval_w_distances(train_data_ref, train_data_pt, independent=False))
                    w_list_eva_pair.append(eval_w_distances(synth_hc,synth_pt, independent=False))
                    if epoch%5==0:
                        predicted_label, predicted_class = label_change(output_eva['logits_x'],opt['ncategories'])
                        #print("")
                        print("class memebership", predicted_class)
                        total_label_change = np.count_nonzero(np.absolute(np.array(predicted_label)-np.array(predicted_label_past))) 
                        print("Total label change ", total_label_change)
                        predicted_label_past = predicted_label
                        cluster_stability.append(total_label_change)
                        cluster_stability.pop(0)
                        if tau <= (0.5+1e-6) and epoch>(min_epochs_for_earlystopping):
                            if (max(cluster_stability)< max(int(cv_data_pt.shape[0]*0.01),10)) and (reconx_loss_eva[-1] > min(reconx_loss_eva[-patience:])+tolerance) and (recony_loss_eva[-1] > min(recony_loss_eva[-patience:])+tolerance):
                                loss_list = {'combined_loss': G_combined_loss, 'recon_x': [reconx_loss, reconx_loss_eva], 'recon_y': [recony_loss, recony_loss_eva], 'kl_loss': [kl_z_loss, kl_scat_loss, kl_scont_loss,kl_scont_loss_y, kl_scontprior_loss_y], 
                                'mmd_loss':mmd_z_loss, 'w_list_hc':[w_list_hc, w_list_eva_hc], 'w_list_pt':[w_list_pt, w_list_eva_pt],'w_list_pair':[w_list_pair, w_list_eva_pair],'tau':tau_list, 'entropy':entropy_list, 'consistency_list':consistency_list}
                                model.save_model(model, epoch, loss_list, opt, path=opt['path'])
                                # model eval
                                logits_eva=output_eva['logits_x'].view(-1,1,opt['ncategories']).to('cpu').numpy() 
                                # for clustering stability we want to evaluate without gumbel sampling
                                clusteringM.append(logits_eva)
                                model.tau=tau
                                return model, clusteringM, loss_list, True
                
                if epoch%50==0:
                    for i in range(8):
                        ax=plt.subplot(2,4,i+1)
                        ax.scatter(cv_data_pt[:,0].to('cpu').numpy(),cv_data_pt[:,i].to('cpu').numpy(), alpha=0.2)
                        ax.scatter(cv_data_ref[:,0].to('cpu').numpy(),cv_data_ref[:,i].to('cpu').numpy(), alpha=0.2)
                        ax.scatter(output_eva['y_recon'][:,0].to('cpu').numpy(),output_eva['y_recon'][:,i].to('cpu').numpy(), alpha=0.2)
                        ax.scatter(output_eva['x_recon'][:,0].to('cpu').numpy(),output_eva['x_recon'][:,i].to('cpu').numpy(), alpha=0.2)
                    plt.legend(['real PT','real HC','fake HC', 'fake PT'])
                    plt.savefig(os.path.join(opt['path'],f'CASLVAEsingledecoder_quality_epoch{epoch}.jpg'))
                    plt.close()
            
            loss_list = {'combined_loss': G_combined_loss, 'recon_x': [reconx_loss, reconx_loss_eva], 'recon_y': [recony_loss, recony_loss_eva], 'kl_loss': [kl_z_loss, kl_scat_loss, kl_scont_loss,kl_scont_loss_y, kl_scontprior_loss_y], 
                            'mmd_loss':mmd_z_loss, 'w_list_hc':[w_list_hc, w_list_eva_hc], 'w_list_pt':[w_list_pt, w_list_eva_pt],'w_list_pair':[w_list_pair, w_list_eva_pair],'tau':tau_list, 'entropy':entropy_list, 'consistency_list':consistency_list}
            
            model.save_model(model, epoch, loss_list, opt, path=opt['path'])
            # save cluster memebership
            logits_eva=output_eva['logits_x'].view(-1,1,opt['ncategories']).to('cpu').numpy() 
            clusteringM.append(logits_eva)
            model.tau=tau
            return model, clusteringM, loss_list, False




