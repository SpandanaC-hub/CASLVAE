# conditional VAE
import os, time, numpy as np, math
from utils import format_time,  weights_init,  dotdict
from evaluate import eval_w_distances , label_change
import torch
import torch.nn.functional as F
import torch.distributions as D
import torch.nn as nn
from networks import CVAE_network
from collections import OrderedDict

import matplotlib.pyplot as plt

class CVAE(object):
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
        self.kl_z_lambda = None

        ##### define scheduler variables
        self.lr_lambda = None
        self.lr_scheduler_G = None
        
        
    def create(self, opt):
        self.opt = dotdict({})
        for key in opt.keys():
            self.opt[key] = opt[key]

        ## define all netwotks
        # Generate control data and discriminate between real and generated
        self.G_forward = CVAE_network(self.opt.nfeatures, self.opt.nLatent_z).to(self.opt.DEVICE)

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
        self.kl_z_lambda=self.opt.kl_z_lambda


    def train_instance(self, real_x, real_y, covariate_x=None, covariate_y=None):
        ## Input to networks
        assert len(real_x.shape)==2 and real_x.shape[-1]==self.opt.nfeatures, 'real_x should be a 2D tensor with last dim equal to number of ROIs/features'
        assert len(real_y.shape)==2 and real_y.shape[-1]==self.opt.nfeatures, 'real_y should be a 2D tensor with last dim equal to number of ROIs/features'

        real_x = real_x.to(self.opt.DEVICE)
        real_y = real_y.to(self.opt.DEVICE)

        ## set networks to train mode
        self.G_forward.train()

        real_data = torch.concat([real_x,real_y],dim=0)
        labels = torch.concat([torch.ones(real_x.shape[0]),torch.zeros(real_y.shape[0])],dim=0).type(torch.int64)
        labels_onehot = F.one_hot(labels, num_classes=2).squeeze().to(self.opt.DEVICE)

        # get latents and reconstructions                
        output = self.G_forward(real_data, labels_onehot) 
        
        mu_z = output['mu_z']
        logsigma_z = output['logsigma_z']
        z = output['z']

        #------------------------------------------------------------------------------------
        # (1) reconstruction loss
        #-------------------------------------------------------------------------------------
        sq_error = self.criterionRecon(output['recon'], real_data, reduction='none').sum(dim=-1)
        recon_loss = sq_error.sum()/real_data.shape[0]
        #-----------------------------------------------------------------------------------------
        #2) KL divergence on  common latent z
        #-----------------------------------------------------------------------------------------
        kldiv = (1 + 2*logsigma_z - torch.square(mu_z) - torch.square(logsigma_z.exp()))
        kldiv = -0.5*torch.sum(kldiv)
        kldiv /= real_data.shape[0]

        loss_G = recon_loss + self.kl_z_lambda*kldiv

        ## update weights of model
        self.optimizer_G.zero_grad()
        loss_G.backward()
        self.optimizer_G.step()

        losses = OrderedDict([('Combined_loss', loss_G.item()), ('kldiv_z', kldiv.item()), ('recon_loss_x', sq_error[labels==1].mean().item()), ('recon_loss_y', sq_error[labels==0].mean().item())])
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


    def save_model(self, model, epoch, loss_list, opt, path='./', checkpointfilename='model_checkpoint_cvae.pt'):
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





def train_CVAE(model, opt, train_loader, train_data_ref, train_data_pt, cv_data_ref, cv_data_pt):
    # initialize variables to store losses
    clusteringM=[]
    kl_z_loss = []
    reconx_loss = []
    recony_loss = []
    G_combined_loss = []
    reconx_loss_eva = []
    recony_loss_eva = []

    w_list_hc=[] # WD between real HC and fake HC
    w_list_eva_hc=[]
    w_list_pt=[]# WD between real PT and fake PT
    w_list_eva_pt=[]
    w_list_pair=[] # WD between real HC and real PT
    w_list_eva_pair=[] # WD between fake HC and fake PT

    n_batches=len(train_loader)
    max_epochs=opt['epochs']
    min_epochs_for_earlystopping = int(max_epochs/2)


    reduce = 'mean'
    patience = 20
    tolerance = 1e-1
    for epoch in range(max_epochs):
        kl_z_batch = 0 
        recon_x_batch = 0
        recon_y_batch = 0
        G_combined_batch_loss = 0
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
            # Progress update every 40 batches.
        #if i % 40 == 0 and not i == 0:
        # Calculate elapsed time in minutes.
        elapsed = format_time(time.time() - t0)
        # Report progress.
        #print('  Batch {:>5,}  of  {:>5,}. \nLosses {}'.format(i, n_batches, losses))

        kl_z_loss.append(kl_z_batch/n_batches) # G adv loss
        reconx_loss.append(recon_x_batch /n_batches) # recon c loss
        G_combined_loss.append(G_combined_batch_loss/n_batches) # combined loss
        recony_loss.append(recon_y_batch/n_batches) # recon y
        print("")
        print("  Average loss: {0:.2f}".format(G_combined_loss[-1]))
        #print("  Average recon x loss: {0:.2f}".format(reconx_loss[-1]))
        #print("  Average recon y loss: {0:.2f}".format(recony_loss[-1]))
        print("  Training epoch took: {:}".format(format_time(time.time() - t0)))

        # model eval
        model.G_forward.eval()

        with torch.no_grad():
            real_data = torch.concat([train_data_pt, train_data_ref],dim=0)
            labels = torch.concat([torch.ones(train_data_pt.shape[0]),torch.zeros(train_data_ref.shape[0])],dim=0).type(torch.int64)
            labels_onehot = F.one_hot(labels, num_classes=2).squeeze().to(opt['DEVICE'])
            real_data_cv = torch.concat([cv_data_pt, cv_data_ref],dim=0)
            labels_cv = torch.concat([torch.ones(cv_data_pt.shape[0]),torch.zeros(cv_data_ref.shape[0])],dim=0).type(torch.int64)
            labels_onehot_cv = F.one_hot(labels_cv, num_classes=2).squeeze().to(opt['DEVICE'])
            output = model.G_forward(real_data, labels_onehot)
            output_eva = model.G_forward(real_data_cv, labels_onehot_cv)

            reconx_loss_eva.append(float(F.mse_loss(output_eva['x_recon'], cv_data_pt, reduction='sum')/cv_data_pt.shape[0]))
            recony_loss_eva.append(float(F.mse_loss(output_eva['y_recon'], cv_data_ref, reduction='sum')/cv_data_ref.shape[0]))

            z_synth = torch.randn(800,opt['nLatent_z'], device='cuda')
            
            synth_hc = model.G_forward.decoder(z_synth[:400], F.one_hot(torch.randint(0,1,size=(400,1)),num_classes=2).squeeze().to(opt['DEVICE']))
            synth_pt = model.G_forward.decoder(z_synth[400:], F.one_hot(torch.randint(1,2,size=(400,1)),num_classes=2).squeeze().to(opt['DEVICE'])) 


            w_list_hc.append(eval_w_distances(train_data_ref, output['y_recon'], independent=False))
            w_list_eva_hc.append(eval_w_distances(cv_data_ref, output_eva['y_recon'],  independent=False))
            w_list_pt.append(eval_w_distances(train_data_pt, output['x_recon'], independent=False))
            w_list_eva_pt.append(eval_w_distances(cv_data_pt, output_eva['x_recon'],  independent=False))
            w_list_pair.append(eval_w_distances(train_data_ref, train_data_pt, independent=False))
            w_list_eva_pair.append(eval_w_distances(synth_hc,synth_pt, independent=False))

            if epoch%5==0:
                if epoch>(min_epochs_for_earlystopping):
                    if (reconx_loss_eva[-1] > min(reconx_loss_eva[-patience:])+tolerance) and (recony_loss_eva[-1] > min(recony_loss_eva[-patience:])+tolerance):
                        loss_list = {'combined_loss': G_combined_loss, 'recon_x': [reconx_loss, reconx_loss_eva], 'recon_y': [recony_loss, recony_loss_eva], 'kl_loss': [kl_z_loss],
                        'w_list_hc':[w_list_hc, w_list_eva_hc], 'w_list_pt':[w_list_pt, w_list_eva_pt],'w_list_pair':[w_list_pair, w_list_eva_pair]}
                        model.save_model(model, epoch, loss_list, opt, path=opt['path'])
                        # model eval
                        zcont_eva=output_eva['mu_z'].to('cpu').numpy() 
                        # return salient latent
                        clusteringM.append(zcont_eva)
                        return model, clusteringM, loss_list, True
        
        if epoch%50==0:
            for i in range(8):
                ax=plt.subplot(2,4,i+1)
                ax.scatter(cv_data_pt[:,0].to('cpu').numpy(),cv_data_pt[:,i].to('cpu').numpy(), alpha=0.2)
                ax.scatter(cv_data_ref[:,0].to('cpu').numpy(),cv_data_ref[:,i].to('cpu').numpy(), alpha=0.2)
                ax.scatter(output_eva['y_recon'][:,0].to('cpu').numpy(),output_eva['y_recon'][:,i].to('cpu').numpy(), alpha=0.2)
                ax.scatter(output_eva['x_recon'][:,0].to('cpu').numpy(),output_eva['x_recon'][:,i].to('cpu').numpy(), alpha=0.2)
            plt.legend(['real PT','real HC','fake HC', 'fake PT'])
            plt.savefig(os.path.join(opt['path'],f'cVAE_quality_epoch{epoch}.jpg'))
            plt.close()

    loss_list = {'combined_loss': G_combined_loss, 'recon_x': [reconx_loss, reconx_loss_eva], 'recon_y': [recony_loss, recony_loss_eva], 'kl_loss': [kl_z_loss],
    'w_list_hc':[w_list_hc, w_list_eva_hc], 'w_list_pt':[w_list_pt, w_list_eva_pt],'w_list_pair':[w_list_pair, w_list_eva_pair]}

    model.save_model(model, epoch, loss_list, opt, path=opt['path'])
    zcont_eva=output_eva['mu_z'].to('cpu').numpy() 
    clusteringM.append(zcont_eva)
    return model, clusteringM, loss_list, False