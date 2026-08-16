import time, numpy as np, os
from utils import sample_gumbel_softmax, format_time
from evaluate import eval_w_distances , label_change
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt



def train_model(model, opt, train_loader, train_data_ref, train_data_pt, cv_data_ref, cv_data_pt, verbose=True, wdist_monitor=False):
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

            norm_reconsalient_list = []
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
                    norm_reconsalient_y_batch += losses['y_recon_s_loss'] # L1 norm on recon salient for y
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
                norm_reconsalient_list.append(norm_reconsalient_y_batch/n_batches)
                kl_scont_loss_y.append(kl_scont_y_batch/n_batches) # d latent 
                kl_scontprior_loss_y.append(kl_scontprior_y_batch/n_batches) # d latent 
                consistency_list.append(consistency_batch/n_batches)
                print("")
                print("  Average loss: {0:.2f}".format(G_combined_loss[-1]))
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
                    
                    synth_hc = model.G_forward.decode_common(z_synth[:400]) +  model.G_forward.decode_salient(scat_preset_synth, scont_preset_synth)
                    synth_pt = model.G_forward.decode_common(z_synth[400:]) +  model.G_forward.decode_salient(scat_synth, scont_synth+mu_sprior_synth)

                    if wdist_monitor:
                        w_list_hc.append(eval_w_distances(train_data_ref, output['y_recon'], independent=False))
                        w_list_eva_hc.append(eval_w_distances(cv_data_ref, output_eva['y_recon'],  independent=False))
                        w_list_pt.append(eval_w_distances(train_data_pt, output['x_recon'], independent=False))
                        w_list_eva_pt.append(eval_w_distances(cv_data_pt, output_eva['x_recon'],  independent=False))
                        w_list_pair.append(eval_w_distances(train_data_ref, train_data_pt, independent=False))
                        w_list_eva_pair.append(eval_w_distances(synth_hc, synth_pt, independent=False))

                    if epoch%5==0:
                        predicted_label, predicted_class = label_change(output_eva['logits_x'],opt['ncategories'])
                        #print("")
                        print("class memebership", predicted_class)
                        total_label_change = np.count_nonzero(np.absolute(np.array(predicted_label)-np.array(predicted_label_past))) 
                        print("Total label change", total_label_change)
                        predicted_label_past = predicted_label
                        cluster_stability.append(total_label_change)
                        cluster_stability.pop(0)
                        if tau <= (0.5+1e-6) and epoch>(min_epochs_for_earlystopping):
                            if (max(cluster_stability)< max(int(cv_data_pt.shape[0]*0.01),10)) and (reconx_loss_eva[-1] > min(reconx_loss_eva[-patience:])+tolerance) and (recony_loss_eva[-1] > min(recony_loss_eva[-patience:])+tolerance):
                                loss_list = {'combined_loss': G_combined_loss, 'recon_x': [reconx_loss, reconx_loss_eva], 'recon_y': [recony_loss, recony_loss_eva], 'kl_loss': [kl_z_loss, kl_scat_loss, kl_scont_loss,kl_scont_loss_y, kl_scontprior_loss_y], 'mmd_loss':mmd_z_loss, 'w_list_hc':[w_list_hc, w_list_eva_hc], 'w_list_pt':[w_list_pt, w_list_eva_pt],'w_list_pair':[w_list_pair, w_list_eva_pair],'tau':tau_list, 'entropy':entropy_list,'norm_reconsalient_y':norm_reconsalient_list,'consistency_list':consistency_list}
                                model.save_model(model, epoch, loss_list, opt, path=opt['path'])
                                # model eval
                                logits_eva=output_eva['logits_x'].view(-1,1,opt['ncategories']).to('cpu').numpy() 
                                # for clustering stability we want to evaluate without gumbel sampling
                                clusteringM.append(logits_eva)
                                model.tau = tau_list[-1]
                                return model, clusteringM, loss_list, True
                
                if (epoch%50==0) and verbose:
                    for i in range(8):
                        ax=plt.subplot(2,4,i+1)
                        ax.scatter(cv_data_pt[:,0].to('cpu').numpy(),cv_data_pt[:,i].to('cpu').numpy(), alpha=0.2)
                        ax.scatter(cv_data_ref[:,0].to('cpu').numpy(),cv_data_ref[:,i].to('cpu').numpy(), alpha=0.2)
                        ax.scatter(output_eva['y_recon'][:,0].to('cpu').numpy(),output_eva['y_recon'][:,i].to('cpu').numpy(), alpha=0.2)
                        ax.scatter(output_eva['x_recon'][:,0].to('cpu').numpy(),output_eva['x_recon'][:,i].to('cpu').numpy(), alpha=0.2)
                    plt.legend(['real PT','real HC','fake HC', 'fake PT'])
                    plt.savefig(os.path.join(opt['path'],f'CASLVAE_quality_epoch{epoch}.jpg'))
                    plt.close()
            
            loss_list = {'combined_loss': G_combined_loss, 'recon_x': [reconx_loss, reconx_loss_eva], 'recon_y': [recony_loss, recony_loss_eva], 'kl_loss': [kl_z_loss, kl_scat_loss, kl_scont_loss,kl_scont_loss_y, kl_scontprior_loss_y],                             'mmd_loss':mmd_z_loss, 'w_list_hc':[w_list_hc, w_list_eva_hc], 'w_list_pt':[w_list_pt, w_list_eva_pt],'w_list_pair':[w_list_pair, w_list_eva_pair],'tau':tau_list, 'entropy':entropy_list,'norm_reconsalient_y':norm_reconsalient_list,'consistency_list':consistency_list}
            
            model.save_model(model, epoch, loss_list, opt, path=opt['path'])
            # save cluster memebership
            logits_eva=output_eva['logits_x'].view(-1,1,opt['ncategories']).to('cpu').numpy() 
            clusteringM.append(logits_eva)
            model.tau = tau_list[-1]
            return model, clusteringM, loss_list, False

