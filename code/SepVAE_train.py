
import os, time, numpy as np, math
from utils import format_time,  weights_init,  dotdict
from evaluate import eval_w_distances , label_change
import torch
import torch.nn.functional as F
import torch.distributions as D
import torch.nn as nn
from networks import SepVAE_network
from collections import OrderedDict

import matplotlib.pyplot as plt


class discriminator_model(nn.Module):
    def __init__(self, nLatent_s_cont, nLatent_z):
        super(discriminator_model, self).__init__()
        inp_dim=nLatent_z+nLatent_s_cont
        self.discriminator=nn.Sequential(nn.Linear(inp_dim,int((inp_dim)/2)),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(int((inp_dim)/2), 1))
    def forward(self, z, s):
        joint = torch.concat([z,s],dim=-1) # z and s latent
        q_score = nn.Sigmoid()(self.discriminator(joint)) # probability of joint samples
        return q_score

class SepVAE(object):
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
        self.G_forward = SepVAE_network(self.opt.nfeatures, self.opt.nLatent_z, self.opt.nLatent_s_cont).to(self.opt.DEVICE)
        self.TCdisc = discriminator_model(self.opt.nLatent_s_cont, self.opt.nLatent_z).to(self.opt.DEVICE)

        ## initiate networks
        self.G_forward.apply(weights_init)
        self.TCdisc.apply(weights_init)

        ## define all optimizers
        self.optimizer_G = torch.optim.Adam(self.G_forward.parameters(),
                                        lr=self.opt.lr, betas=(self.opt.beta1, 0.999),weight_decay=2.5*1e-3)

        self.optimizer_TCdisc = torch.optim.Adam(self.TCdisc.parameters(),
                                        lr=self.opt.lr/5, betas=(self.opt.beta1, 0.999))

        ## define all schedulers
        if self.opt.scheduler:
            self.lr_lambda = DecayLR(self.opt.epochs, 0, self.opt.decay_epochs).step
            self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=self.lr_lambda)

        ## define hyperparameters
        self.kl_scont_lambda=self.opt.kl_scont_lambda
        self.kl_sconty_lambda=self.opt.kl_sconty_lambda
        self.kl_z_lambda=self.opt.kl_z_lambda
        self.tc_lambda = self.opt.tc_lambda
        self.cls_lambda = self.opt.cls_lambda


    def train_instance(self, real_x, real_y, covariate_x=None, covariate_y=None):
        ## Input to networks
        assert len(real_x.shape)==2 and real_x.shape[-1]==self.opt.nfeatures, 'real_x should be a 2D tensor with last dim equal to number of ROIs/features'
        assert len(real_y.shape)==2 and real_y.shape[-1]==self.opt.nfeatures, 'real_y should be a 2D tensor with last dim equal to number of ROIs/features'

        real_x = real_x.to(self.opt.DEVICE)
        real_y = real_y.to(self.opt.DEVICE)

        ## set networks to train mode
        self.G_forward.train()
        self.TCdisc.train()

        # get latents and reconstructions                
        output = self.G_forward(real_x, real_y) 
        
        mu_s_x = output['mu_s_x']
        logsigma_s_x = output['logsigma_s_x']
        mu_s_y = output['mu_s_y']
        logsigma_s_y  = output['logsigma_s_y']
        z_x = output['z_x']
        z_y = output['z_y']
        mu_z_x = output['mu_z_x']
        logsigma_z_x = output['logsigma_z_x']
        mu_z_y = output['mu_z_y']
        logsigma_z_y = output['logsigma_z_y']
        y_recon_s = output['y_recon_s']
        scont_x = output['scont_x']
        ###--------------------------------------
        # Step 1 Optimize discriminator
        ###--------------------------------------
        q_score = self.TCdisc(z_x.detach(),scont_x.detach()) # discriminator output for joint samples
        batch_sizehalf = int(z_x.shape[0]/2)
        qbar_score = self.TCdisc(z_x.detach(), torch.concat([scont_x[batch_sizehalf:],scont_x[:batch_sizehalf]],dim=0).detach()) # discriminator output for shuffled samples

        disc_loss = torch.mean(-torch.log(q_score)-torch.log(1-qbar_score)) #Binary Cross entropy
        self.optimizer_TCdisc.zero_grad()
        disc_loss.backward()
        self.optimizer_TCdisc.step()

        self.TCdisc.eval()
        #---------------------------------------------
        # Step 2
        # Total Correlation Loss: Density ratio trick
        #----------------------------------------------
        q_score = self.TCdisc(z_x,scont_x)
        TC_loss = torch.mean(F.relu(torch.log(q_score)-torch.log(1-q_score)))  
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
        #3) KL divergence on  salient latent s for x samples
        #-----------------------------------------------------------------------------------------
        kldiv_s_cont_x = (1 + 2*logsigma_s_x - torch.square(mu_s_x) - torch.square(logsigma_s_x.exp()))
        kldiv_s_cont_x = -0.5*torch.sum(kldiv_s_cont_x)
        kldiv_s_cont_x /= real_x.shape[0]
        #-----------------------------------------------------------------------------------------
        #4) KL divergence on  salient latent s for y samples
        #-----------------------------------------------------------------------------------------   
        # for y, we want posterior and prior means to be 0, std is preset to sigma_y=0.1
        kldiv_s_cont_y_penalty = 0.5*torch.sum(torch.square(mu_s_y))/real_y.shape[0]
        #-----------------------------------------------------------------------------------------
        #5) Saleint classification loss
        #----------------------------------------------------------------------------------------- 
        cls_loss = torch.mean(-torch.log(output['cls_s_x']) - torch.log(1-output['cls_s_y'])) # BCE loss

        loss_G = recon_loss_x + recon_loss_y + self.kl_z_lambda*kldivxy + self.kl_scont_lambda*(kldiv_s_cont_x) + self.kl_sconty_lambda*(kldiv_s_cont_y_penalty) + self.tc_lambda*TC_loss + self.cls_lambda*cls_loss
        ## update weights of model
        self.optimizer_G.zero_grad()
        loss_G.backward()
        self.optimizer_G.step()

        losses = OrderedDict([('Combined_loss', loss_G.item()), ('kldiv_z', kldivxy.item()), ('recon_loss_x', recon_loss_x.item()), ('recon_loss_y', recon_loss_y.item()),   
                 ('kldiv_s_cont', kldiv_s_cont_x.item()), ('discriminator_loss',disc_loss.item()), ('salientclassification_loss', cls_loss.item()), ('MI_loss', TC_loss.item()),  ('kldiv_s_cont_y', kldiv_s_cont_y_penalty.item())])
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


    def save_model(self, model, epoch, loss_list, opt, path='./', checkpointfilename='model_checkpoint_sepvae.pt'):
        checkpoint = {
            'G_forward': model.G_forward.state_dict(),
            'optimizer_G': model.optimizer_G.state_dict(),
            'TCdisc': model.TCdisc.state_dict(),
            'optimizer_TCdisc': model.optimizer_TCdisc.state_dict(),
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
        self.TCdisc.load_state_dict(checkpoint['TCdisc'])

        # Load optimizer
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        self.optimizer_TCdisc.load_state_dict(checkpoint['optimizer_TCdisc'])

        # Restore preprocessing stats
        self.train_mean = checkpoint.get('train_mean', None)
        self.train_std = checkpoint.get('train_std', None)
        self.covariatecorrection_params = checkpoint.get('covariatecorrection_params', None)

        # Restore metadata
        self.feature_names = checkpoint.get('feature_names', None)
        self.covariate_names = checkpoint.get('covariate_names', None)
        return checkpoint





def train_SepVAE(model, opt, train_loader, train_data_ref, train_data_pt, cv_data_ref, cv_data_pt):
    # initialize variables to store losses
    clusteringM=[]
    kl_z_loss = []
    kl_scont_loss = []
    kl_scont_loss_y = []
    reconx_loss = []
    recony_loss = []
    G_combined_loss = []
    cls_list =[]
    reconx_loss_eva = []
    recony_loss_eva = []

    w_list_hc=[] # WD between real HC and fake HC
    w_list_eva_hc=[]
    w_list_pt=[]# WD between real PT and fake PT
    w_list_eva_pt=[]
    w_list_pair=[] # WD between real HC and real PT
    w_list_eva_pair=[] # WD between fake HC and fake PT

    TC_list = []
    discriminator_loss = []

    n_batches=len(train_loader)
    sigma_y = 0.1
    max_epochs=opt['epochs']
    min_epochs_for_earlystopping = int(max_epochs/2)


    reduce = 'mean'
    predicted_label_past = np.zeros(cv_data_pt.shape[0])
    cluster_stability = [0,0,0,0] # lets monitor over the last 4 evaluations
    patience = 20
    tolerance = 1e-1
    for epoch in range(max_epochs):
        kl_z_batch = 0 
        recon_x_batch = 0
        recon_y_batch = 0
        cls_batch = 0
        G_combined_batch_loss = 0
        kl_scont_batch = 0
        kl_scont_y_batch = 0
        discriminator_batch = 0
        TC_batch = 0
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch+1, opt['epochs']))
        print('Training...')
        t0 = time.time()
        for i,(ref_batch, pt_batch) in enumerate(train_loader):
            ref_batch = ref_batch.to(opt['DEVICE']).type(torch.float32)
            pt_batch = pt_batch.to(opt['DEVICE']).type(torch.float32)
            #train
            losses = model.train_instance(pt_batch, ref_batch)
            kl_z_batch += losses['kldiv_z'] # kl on z
            G_combined_batch_loss += losses['Combined_loss']
            recon_y_batch += losses['recon_loss_y']
            recon_x_batch += losses['recon_loss_x']
            cls_batch += losses['salientclassification_loss'] # kl on disease categorical latent
            kl_scont_batch += losses['kldiv_s_cont'] # kl on disease continuous latent
            TC_batch += losses['MI_loss'] # MMD on z
            kl_scont_y_batch += losses['kldiv_s_cont_y'] # L2 norm on scont for y
            discriminator_batch += losses['discriminator_loss']
            # Progress update every 40 batches.
        #if i % 40 == 0 and not i == 0:
        # Calculate elapsed time in minutes.
        elapsed = format_time(time.time() - t0)
        # Report progress.
        #print('  Batch {:>5,}  of  {:>5,}. \nLosses {}'.format(i, n_batches, losses))

        kl_z_loss.append(kl_z_batch/n_batches) # G adv loss
        cls_list.append(cls_batch/n_batches) # G adv loss
        kl_scont_loss.append(kl_scont_batch/n_batches) # d latent 
        TC_list.append(TC_batch/n_batches) # d latent 
        reconx_loss.append(recon_x_batch /n_batches) # recon c loss
        G_combined_loss.append(G_combined_batch_loss/n_batches) # combined loss
        recony_loss.append(recon_y_batch/n_batches) # recon y
        discriminator_loss.append(discriminator_batch/n_batches)
        kl_scont_loss_y.append(kl_scont_y_batch/n_batches) # d latent 
        print("")
        print("  Average loss: {0:.2f}".format(G_combined_loss[-1]))
        #print("  Average recon x loss: {0:.2f}".format(reconx_loss[-1]))
        #print("  Average recon y loss: {0:.2f}".format(recony_loss[-1]))
        print("  Training epoch took: {:}".format(format_time(time.time() - t0)))

        # model eval
        model.G_forward.eval()
        model.TCdisc.eval()

        with torch.no_grad():
            output = model.G_forward(train_data_pt, train_data_ref )
            output_eva = model.G_forward(cv_data_pt, cv_data_ref )


            reconx_loss_eva.append(float(F.mse_loss(output_eva['x_recon'], cv_data_pt, reduction='sum')/cv_data_pt.shape[0]))
            recony_loss_eva.append(float(F.mse_loss(output_eva['y_recon'], cv_data_ref, reduction='sum')/cv_data_ref.shape[0]))

            z_synth = torch.randn(800,opt['nLatent_z'], device='cuda')
            scont_synth = torch.randn(400,opt['nLatent_s_cont'], device='cuda')
            
            scont_preset_synth =   torch.randn(400,opt['nLatent_s_cont'], device='cuda')*sigma_y

            synth_hc = model.G_forward.decode_common(z_synth[:400]) +  model.G_forward.decode_salient(scont_preset_synth)
            synth_pt = model.G_forward.decode_common(z_synth[400:]) +  model.G_forward.decode_salient(scont_synth)

            '''
            w_list_hc.append(eval_w_distances(train_data_ref, output['y_recon'], independent=False))
            w_list_eva_hc.append(eval_w_distances(cv_data_ref, output_eva['y_recon'],  independent=False))
            w_list_pt.append(eval_w_distances(train_data_pt, output['x_recon'], independent=False))
            w_list_eva_pt.append(eval_w_distances(cv_data_pt, output_eva['x_recon'],  independent=False))
            w_list_pair.append(eval_w_distances(train_data_ref, train_data_pt, independent=False))
            w_list_eva_pair.append(eval_w_distances(synth_hc,synth_pt, independent=False))
            '''
            if epoch%5==0:
                if epoch>(min_epochs_for_earlystopping):
                    if (reconx_loss_eva[-1] > min(reconx_loss_eva[-patience:])+tolerance) and (recony_loss_eva[-1] > min(recony_loss_eva[-patience:])+tolerance):
                        loss_list = {'combined_loss': G_combined_loss, 'recon_x': [reconx_loss, reconx_loss_eva], 'recon_y': [recony_loss, recony_loss_eva], 'kl_loss': [kl_z_loss, kl_scont_loss,kl_scont_loss_y], 
                        'salientclassification_loss':cls_list, 'w_list_hc':[w_list_hc, w_list_eva_hc], 'w_list_pt':[w_list_pt, w_list_eva_pt],'w_list_pair':[w_list_pair, w_list_eva_pair], 'MI':TC_list,
                        'discriminator_loss':discriminator_loss}
                        model.save_model(model, epoch, loss_list, opt, path=opt['path'])
                        # model eval
                        scont_eva=torch.concat([output_eva['mu_s_x'] ,output_eva['mu_s_y'] ],dim=0).to('cpu').numpy() 
                        # return salient latent
                        clusteringM.append(scont_eva)
                        return model, clusteringM, loss_list, True
        
        if epoch%50==0:
            for i in range(8):
                ax=plt.subplot(2,4,i+1)
                ax.scatter(cv_data_pt[:,0].to('cpu').numpy(),cv_data_pt[:,i].to('cpu').numpy(), alpha=0.2)
                ax.scatter(cv_data_ref[:,0].to('cpu').numpy(),cv_data_ref[:,i].to('cpu').numpy(), alpha=0.2)
                ax.scatter(output_eva['y_recon'][:,0].to('cpu').numpy(),output_eva['y_recon'][:,i].to('cpu').numpy(), alpha=0.2)
                ax.scatter(output_eva['x_recon'][:,0].to('cpu').numpy(),output_eva['x_recon'][:,i].to('cpu').numpy(), alpha=0.2)
            plt.legend(['real PT','real HC','fake HC', 'fake PT'])
            plt.savefig(os.path.join(opt['path'],f'SepVAE_quality_epoch{epoch}.jpg'))
            plt.close()

    loss_list = {'combined_loss': G_combined_loss, 'recon_x': [reconx_loss, reconx_loss_eva], 'recon_y': [recony_loss, recony_loss_eva], 'kl_loss': [kl_z_loss, kl_scont_loss,kl_scont_loss_y], 
                        'salientclassification_loss':cls_list, 'w_list_hc':[w_list_hc, w_list_eva_hc], 'w_list_pt':[w_list_pt, w_list_eva_pt],'w_list_pair':[w_list_pair, w_list_eva_pair], 'MI':TC_list,
                        'discriminator_loss':discriminator_loss}

    model.save_model(model, epoch, loss_list, opt, path=opt['path'])
    scont_eva=torch.concat([output_eva['mu_s_x'] ,output_eva['mu_s_y'] ],dim=0).to('cpu').numpy() 
    clusteringM.append(scont_eva)
    return model, clusteringM, loss_list, False



            
