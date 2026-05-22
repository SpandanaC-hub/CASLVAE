
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import sample_gumbel_softmax

# encoder-decoder architecture, with two encoders and two decoders respectively learning common and salient latent variables
class Generator_CASL_twodecoders(nn.Module):
    def __init__(self, input_dim, nLatent_z, nLatent_s_cat , nLatent_s_cont, ncategories):
        super(Generator_CASL_twodecoders, self).__init__()
        self.nLatent_s_cat = nLatent_s_cat
        self.nLatent_s_cont = nLatent_s_cont
        self.nLatent_z = nLatent_z
        self.ncategories = ncategories

        self.enc_z =nn.Sequential(nn.Linear(input_dim,int(input_dim/2)),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(int(input_dim/2),int(input_dim/4)),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(int(input_dim/4), nLatent_z*2))

        self.enc_s = nn.Sequential(nn.Linear(input_dim,int(input_dim/2)),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(int(input_dim/2), int(input_dim/4)),
                                    nn.LeakyReLU(0.2, inplace=True))
        
        self.enc_s_cat = nn.Sequential(nn.Linear(int(input_dim/4),int(input_dim/4)),
                                        nn.LeakyReLU(0.2, inplace=True),
                                        nn.Linear(int(input_dim/4), self.nLatent_s_cat*self.ncategories))
        
        self.enc_s_cont = nn.Sequential(nn.Linear(int(input_dim/4) + (self.nLatent_s_cat*self.ncategories),int(input_dim/4)),
                                        nn.LeakyReLU(0.2, inplace=True),
                                        nn.Linear(int(input_dim/4), self.nLatent_s_cont))

        self.proj_s_cat = nn.Linear(self.nLatent_s_cat*self.ncategories,int(input_dim/16))
        self.proj_s_cont= nn.Linear(self.nLatent_s_cont,int(input_dim/16))
        
        self.prior_s_cont = nn.Sequential(nn.Linear(self.nLatent_s_cat*self.ncategories,int(input_dim/4)),
                                        nn.LeakyReLU(0.2, inplace=True),
                                        nn.Linear(int(input_dim/4), self.nLatent_s_cont))

        self.dec_z = nn.Sequential(nn.Linear(nLatent_z,int(input_dim/4)),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(int(input_dim/4),int(input_dim/2)),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(int(input_dim/2),input_dim))

        self.dec_s = nn.Sequential(nn.Linear(int(input_dim/16)*2,int(input_dim/4)),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(int(input_dim/4),int(input_dim/2)),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(int(input_dim/2), input_dim))

    def gumbel_sample(self, num_samples=1, device="cpu", hard=True):
        """
        Sample new data from the prior p(z) and decode. prior is uniform i.e. uninformative
        """
        # Uniform categorical prior
        logits = torch.zeros(num_samples, self.nLatent_s_cat, self.ncategories, device=device)
        
        # Gumbel-softmax sampling with uniform logits
        z = sample_gumbel_softmax(logits, tau=1.0, hard=hard)  # hard = True, returns one-hot, otherwise returns continuous approximation of one-hot that lies on (ncategories-1) dimensional simplex
        z_flat = z.view(num_samples, self.nLatent_s_cat * self.ncategories)

        return z_flat
 

    def encode_common(self, input_data):
        mu_z, logsigma_z = self.enc_z(input_data).chunk(2,dim=-1)
        return mu_z, logsigma_z

    def encode_salient_cat(self, input_data):
        proj_input = self.enc_s(input_data)
        logits = self.enc_s_cat(proj_input)
        return logits, proj_input

    def decode_common(self, latent_z):
        return self.dec_z(latent_z)

    def decode_salient(self, latent_s_cat, latent_s_cont):
        proj_scat = self.proj_s_cat(latent_s_cat)
        proj_scont = self.proj_s_cont(latent_s_cont)
        return self.dec_s(torch.cat([proj_scat,proj_scont],dim=-1))
    
    def forward(self, input_x, input_y,tau=1.0):
        # input_x is target data
        # input_y is background data
        device  = input_x.device
        N_x = input_x.shape[0] 
        N_y = input_y.shape[0]
        
        # sample salient categorical latent for input_x
        logits_x, proj_input_x = self.encode_salient_cat(input_x)
        logits_x = logits_x.view(-1, self.nLatent_s_cat, self.ncategories)
        s_cat_x = sample_gumbel_softmax(logits_x, tau).view(-1, self.nLatent_s_cat * self.ncategories) # soft sample
        
        # preset categorical latent for input y
        s_cat_preset_y = torch.zeros(N_y, self.ncategories, device=device)
        s_cat_preset_y[:,0] = 1
        
        # sample salient categorical latent for input_y
        logits_y, proj_input_y = self.encode_salient_cat(input_y)
        logits_y = logits_y.view(-1, self.nLatent_s_cat, self.ncategories)    
        s_cat_y = sample_gumbel_softmax(logits_y,tau).view(-1, self.nLatent_s_cat * self.ncategories)
        
        # get posterior distribution of common latent ~N(mu,diag(sigma))
        mu_z_y, logsigma_z_y = self.encode_common(input_y) 
        mu_z_x, logsigma_z_x = self.encode_common(input_x) 
        
        # get prior means of salient continuous latent 
        mu_sprior_x = self.prior_s_cont(s_cat_x)#.chunk(2, dim=-1) 
        mu_sprior_y = self.prior_s_cont(s_cat_preset_y)#.chunk(2, dim=-1) 
        # get posterior means of salient continuous latent 
        mu_s_x = self.enc_s_cont(torch.cat([proj_input_x, s_cat_x],dim=-1)) 
        mu_s_y = self.enc_s_cont(torch.cat([proj_input_y, s_cat_preset_y],dim=-1)) 

        # sample common latent 
        z_x = torch.mul(torch.randn(N_x,self.nLatent_z, device=device), abs(logsigma_z_x.exp())) + mu_z_x
        z_y = torch.mul(torch.randn(N_y,self.nLatent_z, device=device), abs(logsigma_z_y.exp())) + mu_z_y
        
        # sample salient continuous latent 
        s_cont_x = torch.randn(N_x, self.nLatent_s_cont, device=device) + mu_s_x
        s_cont_preset_y = torch.zeros(N_y, self.nLatent_s_cont, device=device) # for y, preset s_cont = 0

        # decode common and salient latent and add the recontructions
        x_recon_z = self.decode_common(z_x)
        y_recon_z = self.decode_common(z_y)
        x_recon_s = self.decode_salient(s_cat_x, s_cont_x)
        y_recon_s = self.decode_salient(s_cat_preset_y, s_cont_preset_y)

        x_recon = x_recon_z + x_recon_s
        y_recon = y_recon_z + y_recon_s

        return {'x_recon': x_recon, 'y_recon': y_recon, 's_cat_x': s_cat_x, 's_cat_y': s_cat_y, 'y_recon_s': y_recon_s, 
                'z_x':z_x, 'z_y':z_y, 'mu_z_x': mu_z_x, 'logsigma_z_x': logsigma_z_x, 'mu_z_y': mu_z_y, 'logsigma_z_y': logsigma_z_y,
                'logits_x': logits_x, 'mu_s_x': mu_s_x, 'mu_s_y': mu_s_y, 'mu_sprior_x': mu_sprior_x, 'mu_sprior_y': mu_sprior_y}


# encoder-decoder architecture, with two encoders and two decoder respectively learning common and salient latent variables - interaction in last layers
class Generator_CASL_singledecoderlastlayer(nn.Module):
    def __init__(self, input_dim, nLatent_z, nLatent_s_cat , nLatent_s_cont, ncategories):
        super(Generator_CASL_singledecoderlastlayer, self).__init__()
        self.nLatent_s_cat = nLatent_s_cat
        self.nLatent_s_cont = nLatent_s_cont
        self.nLatent_z = nLatent_z
        self.ncategories = ncategories

        self.enc_z =nn.Sequential(nn.Linear(input_dim,int(input_dim/2)),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(int(input_dim/2),int(input_dim/4)),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(int(input_dim/4), nLatent_z*2))

        self.enc_s = nn.Sequential(nn.Linear(input_dim,int(input_dim/2)),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(int(input_dim/2), int(input_dim/4)),
                                    nn.LeakyReLU(0.2, inplace=True))
        
        self.enc_s_cat = nn.Sequential(nn.Linear(int(input_dim/4),int(input_dim/4)),
                                        nn.LeakyReLU(0.2, inplace=True),
                                        nn.Linear(int(input_dim/4), self.nLatent_s_cat*self.ncategories))
        
        self.enc_s_cont = nn.Sequential(nn.Linear(int(input_dim/4) + (self.nLatent_s_cat*self.ncategories),int(input_dim/4)),
                                        nn.LeakyReLU(0.2, inplace=True),
                                        nn.Linear(int(input_dim/4), self.nLatent_s_cont))
        self.proj_z = nn.Linear(self.nLatent_z,int(input_dim/16))
        self.proj_s_cat = nn.Linear(self.nLatent_s_cat*self.ncategories,int(input_dim/16))
        self.proj_s_cont= nn.Linear(self.nLatent_s_cont,int(input_dim/16))
        
        self.prior_s_cont = nn.Sequential(nn.Linear(self.nLatent_s_cat*self.ncategories,int(input_dim/4)),
                                        nn.LeakyReLU(0.2, inplace=True),
                                        nn.Linear(int(input_dim/4), self.nLatent_s_cont))

        self.dec_z = nn.Sequential(nn.Linear(int(input_dim/16),int(input_dim/4)),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(int(input_dim/4),int(input_dim/2)),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(int(input_dim/2),input_dim))
                                    
        self.dec_s = nn.Sequential(nn.Linear(int(input_dim/16)*2,int(input_dim/4)),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(int(input_dim/4),int(input_dim/2)),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(int(input_dim/2),input_dim))
        
        self.dec = nn.Sequential(nn.Linear(input_dim,input_dim),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(input_dim,input_dim))

    def gumbel_sample(self, num_samples=1, device="cpu", hard=True):
        """
        Sample new data from the prior p(z) and decode. prior is uniform i.e. uninformative
        """
        # Uniform categorical prior
        logits = torch.zeros(num_samples, self.nLatent_s_cat, self.ncategories, device=device)
        
        # Gumbel-softmax sampling with uniform logits
        z = sample_gumbel_softmax(logits, tau=1.0, hard=hard)  # hard = True, returns one-hot, otherwise returns continuous approximation of one-hot that lies on (ncategories-1) dimensional simplex
        z_flat = z.view(num_samples, self.nLatent_s_cat * self.ncategories)

        return z_flat
 

    def encode_common(self, input_data):
        mu_z, logsigma_z = self.enc_z(input_data).chunk(2,dim=-1)
        return mu_z, logsigma_z

    def encode_salient_cat(self, input_data):
        proj_input = self.enc_s(input_data)
        logits = self.enc_s_cat(proj_input)
        return logits, proj_input

    def decoder(self, latent_z, latent_s_cat, latent_s_cont):
        proj_z = self.proj_z(latent_z)
        proj_scat = self.proj_s_cat(latent_s_cat)
        proj_scont = self.proj_s_cont(latent_s_cont)
        #return self.dec(torch.cat([proj_z, proj_scont],dim=-1))
        dec_z = self.dec_z(proj_z)
        dec_s = self.dec_s(torch.cat([proj_scat, proj_scont],dim=-1))
        return self.dec(dec_z+dec_s)
    
    def forward(self, input_x, input_y,tau=1.0):
        # input_x is target data
        # input_y is background data
        device  = input_x.device
        N_x = input_x.shape[0] 
        N_y = input_y.shape[0]
        
        # sample salient categorical latent for input_x
        logits_x, proj_input_x = self.encode_salient_cat(input_x)
        logits_x = logits_x.view(-1, self.nLatent_s_cat, self.ncategories)
        s_cat_x = sample_gumbel_softmax(logits_x, tau).view(-1, self.nLatent_s_cat * self.ncategories) # soft sample
        
        # preset categorical latent for input y
        s_cat_preset_y = torch.zeros(N_y, self.ncategories, device=device)
        s_cat_preset_y[:,0] = 1
        
        # sample salient categorical latent for input_y
        logits_y, proj_input_y = self.encode_salient_cat(input_y)
        logits_y = logits_y.view(-1, self.nLatent_s_cat, self.ncategories)    
        s_cat_y = sample_gumbel_softmax(logits_y,tau).view(-1, self.nLatent_s_cat * self.ncategories)
        
        # get posterior distribution of common latent ~N(mu,diag(sigma))
        mu_z_y, logsigma_z_y = self.encode_common(input_y) 
        mu_z_x, logsigma_z_x = self.encode_common(input_x) 
        
        # get prior means of salient continuous latent 
        mu_sprior_x = self.prior_s_cont(s_cat_x)#.chunk(2, dim=-1) 
        mu_sprior_y = self.prior_s_cont(s_cat_preset_y)#.chunk(2, dim=-1) 
        # get posterior means of salient continuous latent 
        mu_s_x = self.enc_s_cont(torch.cat([proj_input_x, s_cat_x],dim=-1)) 
        mu_s_y = self.enc_s_cont(torch.cat([proj_input_y, s_cat_preset_y],dim=-1)) 

        # sample common latent 
        z_x = torch.mul(torch.randn(N_x,self.nLatent_z, device=device), abs(logsigma_z_x.exp())) + mu_z_x
        z_y = torch.mul(torch.randn(N_y,self.nLatent_z, device=device), abs(logsigma_z_y.exp())) + mu_z_y
        
        # sample salient continuous latent 
        s_cont_x = mu_s_x  + torch.randn(N_x, self.nLatent_s_cont, device=device) 
        s_cont_preset_y = torch.zeros(N_y, self.nLatent_s_cont, device=device) # for y, preset s_cont = 0

        # decode common and salient latent 
        x_recon = self.decoder(z_x, s_cat_x, s_cont_x)
        y_recon = self.decoder(z_y, s_cat_preset_y, s_cont_preset_y)

        return {'x_recon': x_recon, 'y_recon': y_recon, 's_cat_x': s_cat_x, 's_cat_y': s_cat_y,
                'z_x':z_x, 'z_y':z_y, 'mu_z_x': mu_z_x, 'logsigma_z_x': logsigma_z_x, 'mu_z_y': mu_z_y, 'logsigma_z_y': logsigma_z_y,
                'logits_x': logits_x, 'mu_s_x': mu_s_x, 'mu_s_y': mu_s_y, 'mu_sprior_x': mu_sprior_x, 'mu_sprior_y': mu_sprior_y}

# encoder-decoder architecture, with two encoders and one decoder respectively learning common and salient latent variables
class Generator_CASL_singledecoder(nn.Module):
    def __init__(self, input_dim, nLatent_z, nLatent_s_cat , nLatent_s_cont, ncategories):
        super(Generator_CASL_singledecoder, self).__init__()
        self.nLatent_s_cat = nLatent_s_cat
        self.nLatent_s_cont = nLatent_s_cont
        self.nLatent_z = nLatent_z
        self.ncategories = ncategories

        self.enc_z =nn.Sequential(nn.Linear(input_dim,int(input_dim/2)),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(int(input_dim/2),int(input_dim/4)),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(int(input_dim/4), nLatent_z*2))

        self.enc_s = nn.Sequential(nn.Linear(input_dim,int(input_dim/2)),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(int(input_dim/2), int(input_dim/4)),
                                    nn.LeakyReLU(0.2, inplace=True))
        
        self.enc_s_cat = nn.Sequential(nn.Linear(int(input_dim/4),int(input_dim/4)),
                                        nn.LeakyReLU(0.2, inplace=True),
                                        nn.Linear(int(input_dim/4), self.nLatent_s_cat*self.ncategories))
        
        self.enc_s_cont = nn.Sequential(nn.Linear(int(input_dim/4) + (self.nLatent_s_cat*self.ncategories),int(input_dim/4)),
                                        nn.LeakyReLU(0.2, inplace=True),
                                        nn.Linear(int(input_dim/4), self.nLatent_s_cont))
        self.proj_z = nn.Sequential(nn.Linear(self.nLatent_z,int(input_dim/16)))
        self.proj_s_cat =  nn.Sequential(nn.Linear(self.nLatent_s_cat*self.ncategories,int(input_dim/16)))
        self.proj_s_cont =  nn.Sequential(nn.Linear(self.nLatent_s_cont,int(input_dim/16)))

        self.prior_s_cont = nn.Sequential(nn.Linear(self.nLatent_s_cat*self.ncategories,int(input_dim/4)),
                                        nn.LeakyReLU(0.2, inplace=True),
                                        nn.Linear(int(input_dim/4), self.nLatent_s_cont))

        self.dec_z = nn.Sequential(nn.Linear(int(input_dim/16),int(input_dim/16)),nn.LeakyReLU(0.2, inplace=True),nn.Linear(int(input_dim/16),int(input_dim/16)))
                                    
        self.dec_s = nn.Sequential(nn.Linear(int(input_dim/16)*2,int(input_dim/16)),nn.LeakyReLU(0.2, inplace=True),nn.Linear(int(input_dim/16),int(input_dim/16)))
        
        dimf = int(input_dim/16)*2#nLatent_z+ ncategories + nLatent_s_cont#
        
        self.dec = nn.Sequential(nn.Linear(dimf,int(input_dim/4)),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(int(input_dim/4),int(input_dim/2)),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(int(input_dim/2), input_dim))

    def gumbel_sample(self, num_samples=1, device="cpu", hard=True):
        """
        Sample new data from the prior p(z) and decode. prior is uniform i.e. uninformative
        """
        # Uniform categorical prior
        logits = torch.zeros(num_samples, self.nLatent_s_cat, self.ncategories, device=device)
        
        # Gumbel-softmax sampling with uniform logits
        z = sample_gumbel_softmax(logits, tau=1.0, hard=hard)  # hard = True, returns one-hot, otherwise returns continuous approximation of one-hot that lies on (ncategories-1) dimensional simplex
        z_flat = z.view(num_samples, self.nLatent_s_cat * self.ncategories)

        return z_flat
 

    def encode_common(self, input_data):
        mu_z, logsigma_z = self.enc_z(input_data).chunk(2,dim=-1)
        return mu_z, logsigma_z

    def encode_salient_cat(self, input_data):
        proj_input = self.enc_s(input_data)
        logits = self.enc_s_cat(proj_input)
        return logits, proj_input

    def decoder(self, latent_z, latent_s_cat, latent_s_cont):
        proj_z = self.proj_z(latent_z)
        proj_scat = self.proj_s_cat(latent_s_cat)
        proj_scont = self.proj_s_cont(latent_s_cont)#self.proj_s(torch.cat([ latent_s_cat, latent_s_cont],dim=-1))#self.proj_s_cont(latent_s_cont)
        #return self.dec(torch.cat([proj_z, proj_scont],dim=-1))
        dec_z = self.dec_z(proj_z)
        dec_s = self.dec_s(torch.cat([proj_scat, proj_scont],dim=-1))
        return self.dec(torch.cat([dec_z,dec_s],dim=-1))#self.dec(torch.cat([dec_z, dec_s],dim=-1))#self.dec(torch.cat([latent_z, latent_s_cat, latent_s_cont],dim=-1))#self.dec(torch.cat([proj_z,dec_s],dim=-1))
    
    def forward(self, input_x, input_y,tau=1.0):
        # input_x is target data
        # input_y is background data
        device  = input_x.device
        N_x = input_x.shape[0] 
        N_y = input_y.shape[0]
        
        # sample salient categorical latent for input_x
        logits_x, proj_input_x = self.encode_salient_cat(input_x)
        logits_x = logits_x.view(-1, self.nLatent_s_cat, self.ncategories)
        s_cat_x = sample_gumbel_softmax(logits_x, tau).view(-1, self.nLatent_s_cat * self.ncategories) # soft sample
        
        # preset categorical latent for input y
        s_cat_preset_y = torch.zeros(N_y, self.ncategories, device=device)
        s_cat_preset_y[:,0] = 1
        
        # sample salient categorical latent for input_y
        logits_y, proj_input_y = self.encode_salient_cat(input_y)
        logits_y = logits_y.view(-1, self.nLatent_s_cat, self.ncategories)    
        s_cat_y = sample_gumbel_softmax(logits_y,tau).view(-1, self.nLatent_s_cat * self.ncategories)
        
        # get posterior distribution of common latent ~N(mu,diag(sigma))
        mu_z_y, logsigma_z_y = self.encode_common(input_y) 
        mu_z_x, logsigma_z_x = self.encode_common(input_x) 
        
        # get prior means of salient continuous latent 
        mu_sprior_x = self.prior_s_cont(s_cat_x)#.chunk(2, dim=-1) 
        mu_sprior_y = self.prior_s_cont(s_cat_preset_y)#.chunk(2, dim=-1) 
        # get posterior means of salient continuous latent 
        mu_s_x = self.enc_s_cont(torch.cat([proj_input_x, s_cat_x],dim=-1)) 
        mu_s_y = self.enc_s_cont(torch.cat([proj_input_y, s_cat_preset_y],dim=-1)) 

        # sample common latent 
        z_x = torch.mul(torch.randn(N_x,self.nLatent_z, device=device), abs(logsigma_z_x.exp())) + mu_z_x
        z_y = torch.mul(torch.randn(N_y,self.nLatent_z, device=device), abs(logsigma_z_y.exp())) + mu_z_y
        
        # sample salient continuous latent 
        s_cont_x = mu_s_x  + torch.randn(N_x, self.nLatent_s_cont, device=device) 
        s_cont_preset_y = torch.zeros(N_y, self.nLatent_s_cont, device=device) # for y, preset s_cont = 0

        # decode common and salient latent 
        x_recon = self.decoder(z_x, s_cat_x, s_cont_x)
        y_recon = self.decoder(z_y, s_cat_preset_y, s_cont_preset_y)

        return {'x_recon': x_recon, 'y_recon': y_recon, 's_cat_x': s_cat_x, 's_cat_y': s_cat_y,
                'z_x':z_x, 'z_y':z_y, 'mu_z_x': mu_z_x, 'logsigma_z_x': logsigma_z_x, 'mu_z_y': mu_z_y, 'logsigma_z_y': logsigma_z_y,
                'logits_x': logits_x, 'mu_s_x': mu_s_x, 'mu_s_y': mu_s_y, 'mu_sprior_x': mu_sprior_x, 'mu_sprior_y': mu_sprior_y}
    


# GMVAE_network follwoing "A NOTE ON DEEP VARIATIONAL MODELS FOR UNSUPERVISED CLUSTERING"
class GMVAE_network(nn.Module):
    def __init__(self, input_dim, nLatent_cat , nLatent_cont, ncategories):
        super(GMVAE_network, self).__init__()
        self.nLatent_cat = nLatent_cat
        self.nLatent_cont = nLatent_cont
        self.ncategories = ncategories

        self.enc_inp_cat = nn.Sequential(nn.Linear(input_dim,int(input_dim/2)),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(int(input_dim/2), int(input_dim/4)),
                                    nn.LeakyReLU(0.2, inplace=True))

        self.enc_inp_cont = nn.Sequential(nn.Linear(input_dim,int(input_dim/2)),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(int(input_dim/2), int(input_dim/4)),
                                    nn.LeakyReLU(0.2, inplace=True))
        
        self.enc_cat = nn.Sequential(nn.Linear(int(input_dim/4),int(input_dim/4)),
                                        nn.LeakyReLU(0.2, inplace=True),
                                        nn.Linear(int(input_dim/4), self.nLatent_cat*self.ncategories))
        
        self.enc_cont = nn.Sequential(nn.Linear(int(input_dim/4) + (self.nLatent_cat*self.ncategories),int(input_dim/4)),
                                        nn.LeakyReLU(0.2, inplace=True),
                                        nn.Linear(int(input_dim/4), self.nLatent_cont))
        
        self.prior_cont = nn.Sequential(nn.Linear(self.nLatent_cat*self.ncategories,int(input_dim/4)),
                                        nn.LeakyReLU(0.2, inplace=True),
                                        nn.Linear(int(input_dim/4), self.nLatent_cont))

        self.dec = nn.Sequential(nn.Linear(nLatent_cont,int(input_dim/4)),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(int(input_dim/4),int(input_dim/2)),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(int(input_dim/2),input_dim))

    
    def gaussian_sample(self, s_mean, s_logsigma):
        # clamp logvar to avoid overflow 
        s_logsigma = torch.clamp(s_logsigma, -10, 10) 
        # compute std safely 
        s_std = torch.exp(s_logsigma)
        eps = torch.randn_like(s_std)
        s = s_mean + eps*s_std
        return s

    def Q_cat(self, input_data):
        proj_input = self.enc_inp_cat(input_data)
        logits = self.enc_cat(proj_input).view(-1, self.nLatent_cat, self.ncategories)
        cat = F.softmax(logits,dim=-1).view(-1, self.nLatent_cat*self.ncategories)
        return logits, cat
        
    def Q_cont(self, input_data, cat):
        proj_input = self.enc_inp_cont(input_data)
        #mu_s, logsigma_s = self.enc_cont(torch.concat([proj_input, cat],dim=-1)).chunk(2,dim=-1)
        mu_s  = self.enc_cont(torch.concat([proj_input, cat],dim=-1))

        # clamp logsigma here 
        logsigma_s = torch.zeros_like(mu_s )#torch.clamp(logsigma_s, -10, 10)
        samples = self.gaussian_sample(mu_s,logsigma_s)
        return mu_s, logsigma_s, samples

    def decoder(self, scont_samples, cat):
        mu_sprior  = self.prior_cont(cat)# get prior distribution params
        #mu_sprior, logsigma_sprior = self.prior_cont(cat).chunk(2,dim=-1) # get prior distribution params
        x_hat = self.dec(scont_samples) # get reconstructed output
        # clamp prior logsigma
        logsigma_sprior = torch.zeros_like(mu_sprior)#torch.clamp(logsigma_sprior, -10, 10)
        return mu_sprior, logsigma_sprior, x_hat
    
    def forward(self, input_x):
        # input_x is data
        device  = input_x.device
        N_x = input_x.shape[0] 
        
        # salient categorical latent for input_x
        logits_x, cat_x = self.Q_cat(input_x)
                
        cat_ = torch.zeros([N_x, self.ncategories]).to(device)

        s, sm, sv, sm_prior, sv_prior, px = [[None] * self.ncategories for i in range(6)]
        # get mixture means
        for i in range(self.ncategories):
            cat = cat_ + torch.eye(self.ncategories).to(device)[i]
            sm[i], sv[i], s[i] = self.Q_cont(input_x, cat)
            sm_prior[i], sv_prior[i], px[i] = self.decoder(s[i], cat)

        # get inference
        with torch.no_grad():
            # Inference for x_recon:
            cat_hat = torch.argmax(cat_x, dim=-1)
            cat_temp = torch.zeros(N_x, self.ncategories)
            cat_temp = torch.scatter(cat_, 1, cat_hat.unsqueeze(1), 1)
            mu_x, logsigma_x, scont_hat = self.Q_cont(input_x, cat_temp)
            mu_prior_x, logsigma_prior_x, x_recon = self.decoder(scont_hat, cat_temp)

        out_train = {'x_recon': px, 's_cat_x': cat_x, 'logits_x': logits_x, 's_cont_x': s, 'mu_s_x': sm,   'logsigma_s_x': sv, 'mu_sprior_x': sm_prior, 'logsigma_sprior_x': sv_prior}
        out_infer = {'x_recon': x_recon, 's_cat_x': cat_x, 'logits_x': logits_x, 's_cont_x': scont_hat, 'mu_s_x': mu_x,   'logsigma_s_x': logsigma_x, 'mu_sprior_x': mu_prior_x, 'logsigma_sprior_x': logsigma_prior_x}
        return out_train, out_infer



#SepVAE_network  
class SepVAE_network(nn.Module):
    def __init__(self, input_dim, nLatent_z, nLatent_s_cont, sigma_y=0.1):
        super(SepVAE_network, self).__init__()
        self.nLatent_s_cont = nLatent_s_cont
        self.nLatent_z = nLatent_z
        self.sigma_y = sigma_y

        self.enc_z =nn.Sequential(nn.Linear(input_dim,int(input_dim/2)),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(int(input_dim/2),int(input_dim/4)),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(int(input_dim/4), nLatent_z*2))

        self.enc_s = nn.Sequential(nn.Linear(input_dim,int(input_dim/2)),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(int(input_dim/2), int(input_dim/4)),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(int(input_dim/4), nLatent_s_cont*2))


        self.dec_z = nn.Sequential(nn.Linear(nLatent_z,int(input_dim/4)),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(int(input_dim/4),int(input_dim/2)),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(int(input_dim/2),input_dim))

        self.dec_s = nn.Sequential(nn.Linear(nLatent_s_cont,int(input_dim/4)),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(int(input_dim/4),int(input_dim/2)),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(int(input_dim/2), input_dim))

        self.classifier = nn.Sequential(nn.Linear(self.nLatent_s_cont,int(self.nLatent_s_cont/2)),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(int(self.nLatent_s_cont/2), 1),
                                    nn.Sigmoid())

    def gaussian_sample(self, mean, logsigma):
        # compute std safely 
        std = torch.exp(logsigma)
        eps = torch.randn_like(std)
        samples = mean + eps*std
        return samples

    def encode_common(self, input_data):
        mu_z, logsigma_z = self.enc_z(input_data).chunk(2,dim=-1)
        return mu_z, logsigma_z

    def encode_salient(self, input_data):
        mu_s, logsigma_s  = self.enc_s(input_data).chunk(2,dim=-1)
        return mu_s, logsigma_s

    def decode_common(self, latent_z):
        return self.dec_z(latent_z)

    def decode_salient(self, latent_s):
        return self.dec_s(latent_s)
    
    def forward(self, input_x, input_y):
        # input_x is target data
        # input_y is background data
        device  = input_x.device
        N_x = input_x.shape[0] 
        N_y = input_y.shape[0]
        
        # get posterior distribution of common latent ~N(mu,diag(sigma))
        mu_z_y, logsigma_z_y = self.encode_common(input_y) 
        mu_z_x, logsigma_z_x = self.encode_common(input_x) 

        # get posterior means of salient continuous latent 
        mu_s_x, logsigma_s_x = self.encode_salient(input_x)
        mu_s_y, logsigma_s_y = self.encode_salient(input_y)
        specific_logsigma_s_y = torch.log(self.sigma_y*torch.ones_like(logsigma_s_y)) # preset posterior std for y

        # sample common latent 
        z_x = self.gaussian_sample(mu_z_x, logsigma_z_x)
        z_y = self.gaussian_sample(mu_z_y, logsigma_z_y)
        
        # sample salient continuous latent 
        s_cont_x = self.gaussian_sample(mu_s_x, logsigma_s_x)
        s_cont_preset_y = self.gaussian_sample(mu_s_y, specific_logsigma_s_y) # for y, s_cont based on preset posterior std

        # decode common and salient latent and add the recontructions
        x_recon_z = self.decode_common(z_x)
        y_recon_z = self.decode_common(z_y)
        x_recon_s = self.decode_salient(s_cont_x)
        y_recon_s = self.decode_salient(s_cont_preset_y)

        x_recon = x_recon_z + x_recon_s
        y_recon = y_recon_z + y_recon_s

        cls_s_x = self.classifier(s_cont_x)
        cls_s_y = self.classifier(s_cont_preset_y)


        return {'x_recon': x_recon, 'y_recon': y_recon, 'y_recon_s': y_recon_s, 'z_x':z_x, 'z_y':z_y, 'cls_s_x': cls_s_x, 'cls_s_y': cls_s_y,
                'mu_z_x': mu_z_x, 'logsigma_z_x': logsigma_z_x, 'mu_z_y': mu_z_y, 'logsigma_z_y': logsigma_z_y,  'scont_x':s_cont_x, 'scont_y':s_cont_preset_y,
                'mu_s_x': mu_s_x, 'mu_s_y': mu_s_y, 'logsigma_s_x': logsigma_s_x, 'logsigma_s_y': specific_logsigma_s_y}




#Conditional VAE network  
class CVAE_network(nn.Module):
    def __init__(self, input_dim, nLatent_z):
        super(CVAE_network, self).__init__()
        self.nLatent_z = nLatent_z
        self.label_dim = 2 # one hot encode binary class label

        self.enc_z =nn.Sequential(nn.Linear(input_dim+self.label_dim,int(input_dim/2)),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(int(input_dim/2),int(input_dim/4)),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(int(input_dim/4), nLatent_z*2))

        self.dec_z = nn.Sequential(nn.Linear(nLatent_z+self.label_dim,int(input_dim/4)),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(int(input_dim/4),int(input_dim/2)),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(int(input_dim/2),input_dim))

    def gaussian_sample(self, mean, logsigma):
        # compute std safely 
        std = torch.exp(logsigma)
        eps = torch.randn_like(std)
        samples = mean + eps*std
        return samples

    def encoder(self, input_data, label):
        mu_z, logsigma_z = self.enc_z(torch.concat([input_data, label],dim=-1)).chunk(2,dim=-1)
        return mu_z, logsigma_z

    def decoder(self, latent_z, label):
        return self.dec_z(torch.concat([latent_z, label],dim=-1))
    
    def forward(self, input_data, label):
        # input data target or backgroudn or both with one hot label where index 0 is background
        device  = input_data.device
        N = input_data.shape[0]
        
        # get posterior distribution of latent ~N(mu,diag(sigma))
        mu_z, logsigma_z = self.encoder(input_data, label) 

        # sample latent 
        z = self.gaussian_sample(mu_z, logsigma_z)
        
        # decode latent
        recon = self.decoder(z, label)
        x_recon = recon[label[:,1]==1]
        y_recon = recon[label[:,0]==1]

        return {'recon': recon,'x_recon': x_recon,'y_recon': y_recon, 'z':z, 'mu_z': mu_z, 'logsigma_z': logsigma_z}
    

#MM-cVAE_network  
class MMcVAE_network(nn.Module):
    def __init__(self, input_dim, nLatent_z, nLatent_s_cont ):
        super(MMcVAE_network, self).__init__()
        self.nLatent_s_cont = nLatent_s_cont
        self.nLatent_z = nLatent_z

        self.enc_z =nn.Sequential(nn.Linear(input_dim,int(input_dim/2)),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(int(input_dim/2),int(input_dim/4)),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(int(input_dim/4), nLatent_z*2))

        self.enc_s = nn.Sequential(nn.Linear(input_dim,int(input_dim/2)),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(int(input_dim/2), int(input_dim/4)),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(int(input_dim/4), nLatent_s_cont*2))


        self.dec_z = nn.Sequential(nn.Linear(nLatent_z,int(input_dim/4)),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(int(input_dim/4),int(input_dim/2)),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(int(input_dim/2),input_dim))

        self.dec_s = nn.Sequential(nn.Linear(nLatent_s_cont,int(input_dim/4)),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(int(input_dim/4),int(input_dim/2)),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(int(input_dim/2), input_dim))

    def gaussian_sample(self, mean, logsigma):
        # compute std safely 
        std = torch.exp(logsigma)
        eps = torch.randn_like(std)
        samples = mean + eps*std
        return samples

    def encode_common(self, input_data):
        mu_z, logsigma_z = self.enc_z(input_data).chunk(2,dim=-1)
        return mu_z, logsigma_z

    def encode_salient(self, input_data):
        mu_s, logsigma_s  = self.enc_s(input_data).chunk(2,dim=-1)
        return mu_s, logsigma_s

    def decode_common(self, latent_z):
        return self.dec_z(latent_z)

    def decode_salient(self, latent_s):
        return self.dec_s(latent_s)
    
    def forward(self, input_x, input_y):
        # input_x is target data
        # input_y is background data
        device  = input_x.device
        N_x = input_x.shape[0] 
        N_y = input_y.shape[0]
        
        # get posterior distribution of common latent ~N(mu,diag(sigma))
        mu_z_y, logsigma_z_y = self.encode_common(input_y) 
        mu_z_x, logsigma_z_x = self.encode_common(input_x) 

        # get posterior means of salient continuous latent 
        mu_s_x, logsigma_s_x = self.encode_salient(input_x)
        mu_s_y, logsigma_s_y = self.encode_salient(input_y)

        # sample common latent 
        z_x = self.gaussian_sample(mu_z_x, logsigma_z_x)
        z_y = self.gaussian_sample(mu_z_y, logsigma_z_y)
        
        # sample salient continuous latent 
        s_cont_x = self.gaussian_sample(mu_s_x, logsigma_s_x)
        s_cont_y = self.gaussian_sample(mu_s_y, logsigma_s_y)

        s_cont_preset_y = torch.zeros(N_y, self.nLatent_s_cont,device=device) # for y, s is preset to direc delta at s'=0


        # decode common and salient latent and add the recontructions
        x_recon_z = self.decode_common(z_x)
        y_recon_z = self.decode_common(z_y)
        x_recon_s = self.decode_salient(s_cont_x)
        y_recon_s = self.decode_salient(s_cont_preset_y)

        x_recon = x_recon_z + x_recon_s
        y_recon = y_recon_z + y_recon_s


        return {'x_recon': x_recon, 'y_recon': y_recon, 'y_recon_s': y_recon_s, 'z_x':z_x, 'z_y':z_y,
                'mu_z_x': mu_z_x, 'logsigma_z_x': logsigma_z_x, 'mu_z_y': mu_z_y, 'logsigma_z_y': logsigma_z_y,  'scont_x':s_cont_x, 'scont_y':s_cont_y,
                'mu_s_x': mu_s_x, 'mu_s_y': mu_s_y, 'logsigma_s_x': logsigma_s_x, 'logsigma_s_y': logsigma_s_y}



