#! ~/anaconda3/envs/NIPS/bin/python3
###PBS -lselect=1:ncpus=4:mem=24gb:ngpus=1:gpu_type=P100 -lwalltime=15:00:00
#PBS -lselect=1:ncpus=4:mem=96gb:ngpus=1:gpu_type=P1000 -lwalltime=15:00:00
__doc__ = """
Example training script with PyTorch. Here's what you need to do. 

Before you run this script, ensure that the following environment variables are set:
    1. AICROWD_OUTPUT_PATH (default: './scratch/shared')
    2. AICROWD_EVALUATION_NAME (default: 'experiment_name')
    3. AICROWD_DATASET_NAME (default: 'cars3d')
    4. DISENTANGLEMENT_LIB_DATA (you may set this to './scratch/dataset' if that's 
                                 where the data lives)

We provide utility functions to make the data and model logistics painless. 
But this assumes that you have set the above variables correctly.    

Once you're done with training, you'll need to export the function that returns
the representations (which we evaluate). This function should take as an input a batch of 
images (NCHW) and return a batch of vectors (NC), where N is the batch-size, C is the 
number of channels, H and W are height and width respectively. 

To help you with that, we provide an `export_model` function in utils_pytorch.py. If your 
representation function is a torch.jit.ScriptModule, you're all set 
(just call `export_model(model)`); if not, it will be traced (!) and the resulting ScriptModule 
will be written out. To learn what tracing entails: 
https://pytorch.org/docs/stable/jit.html#torch.jit.trace 

You'll find a few more utility functions in utils_pytorch.py for pytorch related stuff and 
for data logistics.
"""

import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from l0_layers import L0Dense, L0Pair
import utils_pytorch as pyu

import aicrowd_helpers
import time
import math

parser = argparse.ArgumentParser(description='VAE Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=500, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--pruning', action='store_true', default=False,
                    help='enables Pruning')
parser.add_argument('--model',action='store',default="VAE")
parser.add_argument('--dim',type=int, default=10,metavar='N',
                    help='dimension of latent z')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = pyu.get_loader(batch_size=args.batch_size, **kwargs)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, z_dim):
        super(UnFlatten, self).__init__()
        self.z_dim = z_dim
    def forward(self, input, size=1024):
        return input.view(input.size(0), self.z_dim, 1, 1)


# class Encoder(nn.Module):
#     def __init__(self):
#         super(Encoder, self).__init__()
#         self.tail = nn.Sequential(nn.Linear(4096 * 3, 400),
#                                   nn.ReLU())
#         self.head_mu = nn.Linear(400, 20)
#         self.head_logvar = nn.Linear(400, 20)
#
#     def forward(self, x):
#         h = self.tail(x.contiguous().view(-1, 4096 * 3))
#         return self.head_mu(h), self.head_logvar(h)
#
#
# class Decoder(nn.Sequential):
#     def __init__(self):
#         super(Decoder, self).__init__(nn.Linear(20, 400),
#                                       nn.ReLU(),
#                                       nn.Linear(400, 4096 * 3),
#                                       nn.Sigmoid())

class Encoder(nn.Module):
    def __init__(self,z_dim):
        super(Encoder, self).__init__()
        self.tail = nn.Sequential(nn.Linear(4096 * 3, 400),
                                  nn.ReLU())
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            Flatten(),
            nn.Linear(576, 256),
        )
        if args.pruning != True:
            self.head_mu = nn.Linear(256, z_dim)
            self.head_logvar = nn.Linear(256, z_dim)
        else:
            self.head = L0Pair(256, z_dim, droprate_init=0.2, weight_decay=0.001, lamba=0.1)

    def forward(self, x):
        h = self.encoder(x)
        if args.pruning ==True:
            mu,var = self.head(h)
            logvar = torch.log(var)
            l0_reg = self.head.regularization().cuda()/50000
            mask = self.head.sample_mask()
            return mu, logvar, mask, l0_reg
        else:
            return self.head_mu(h), self.head_logvar(h)

class Decoder(nn.Sequential):
    def __init__(self,z_dim):
        super(Decoder, self).__init__()
        self.decoder1 = nn.Sequential(
            #UnFlatten(z_dim),
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1024))

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding = 1),
            nn.Sigmoid(),
            )
    def forward(self,x):
        output = self.decoder1(x)
        output = self.decoder2(output.reshape(-1, 64, 4, 4))
        return output


class RepresentationExtractor(nn.Module):
    VALID_MODES = ['mean', 'sample']

    def __init__(self, encoder, mode='mean'):
        super(RepresentationExtractor, self).__init__()
       # assert mode in self.VALID_MODES, f'`mode` must be one of {self.VALID_MODES}'
        self.encoder = encoder.cuda()
        self.mode = mode

    def forward(self, x):
        x = x.to(device)
        if args.pruning ==True:
            mu, logvar, mask, l0_reg = self.encoder(x)
        else:
            mu, logvar = self.encoder(x)
        if self.mode == 'mean':
            return mu
        elif self.mode == 'sample':
            return self.reparameterize(mu, logvar)
        else:
            raise NotImplementedError

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

class VAE(nn.Module):
    def __init__(self, z_dim=10):
        super(VAE, self).__init__()
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)

    def forward(self, x):
        if args.pruning ==True:
            mu, logvar, mask,l0_reg = self.encoder(x)
            z = RepresentationExtractor.reparameterize(mu, logvar)
            return self.decoder(z), mu, logvar, z, mask, l0_reg
        else:
            mu, logvar = self.encoder(x)
            z = RepresentationExtractor.reparameterize(mu, logvar)
            return self.decoder(z), mu, logvar, z

z_dim=args.dim
model = VAE(z_dim=z_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)###try SGD!!

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, mask=torch.tensor(1.0)):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 4096 * 3), reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    mask = mask.cuda()        
    KLD = -0.5 * torch.sum(mask.reshape(1,-1)*(1 + logvar - mu.pow(2) - logvar.exp()))

    return BCE + KLD

def gaussian_log_density(samples, mean, log_var):
    normalization = math.log(torch.tensor(2. * math.pi))
    inv_sigma = torch.exp(-log_var)
    tmp = (samples - mean)
    return -0.5 * (tmp * tmp * inv_sigma + log_var + normalization)


def total_correlation(z, z_mean, z_logvar):
    log_qz_prob = gaussian_log_density(torch.unsqueeze(z, 1), torch.unsqueeze(z_mean, 0),torch.unsqueeze(z_logvar, 0))
    log_qz_product = log_sum_exp(log_qz_prob, dim=1).sum(dim=1)
    log_qz = log_sum_exp(log_qz_prob.sum(dim=2),dim=1)
    return (log_qz - log_qz_product).mean()

def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    # TODO: torch.max(value, dim=None) threw an error at time of writing
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


def BetaTC_reg(z_mean, z_logvar, z_sampled, beta=6):
    tc = (beta - 1.) * total_correlation(z_sampled, z_mean, z_logvar)
    return tc


def DIP_loss(mean,lambda_od=20,lambda_d=2):
    exp_mu = torch.mean(mean, dim=0)  #####mean through batch
    # expectation of mu mu.tranpose
    mu_expand1 = mean.unsqueeze(1)  #####(batch_size, 1, number of mean of latent variables)
    mu_expand2 = mean.unsqueeze(
        2)  #####(batch_size, number of mean of latent variables, 1) ignore batch_size, only transpose the means
    exp_mu_mu_t = torch.mean(mu_expand1 * mu_expand2, dim=0)
    # covariance of model mean
    cov = exp_mu_mu_t - exp_mu.unsqueeze(0) * exp_mu.unsqueeze(1)  ##1, mean* mean, 1
    diag_part = torch.diagonal(cov, offset=0, dim1=-2, dim2=-1)
    off_diag_part = cov - torch.diag(diag_part)

    regulariser_od = lambda_od * torch.sum(off_diag_part ** 2)
    regulariser_d = lambda_d * torch.sum((diag_part - 1) ** 2)

    DIP = regulariser_d + regulariser_od

    return DIP



def train(epoch):
    model.train()
    train_loss = 0
    l0_reg=0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device).float()
        optimizer.zero_grad()
        if args.pruning:
            recon_batch, mu, logvar, z_sample,mask, l0_reg = model(data)
            VAE_loss = loss_function(recon_batch, data, mu, logvar, mask)
            loss = VAE_loss+l0_reg
        else:
            recon_batch, mu, logvar, z_sample = model(data)
            VAE_loss = loss_function(recon_batch, data, mu, logvar)
            loss = VAE_loss
        """
        BetaTC = BetaTC_reg(mu,logvar,z_sample)
        loss = loss_function(recon_batch, data, mu, logvar)+BetaTC
        """
        DIP = DIP_loss(mu)
        loss = loss+DIP
        loss.backward() 
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tDIP_Loss:{:.6f}\tL0_Loss:{:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                VAE_loss.item() / len(data), DIP,l0_reg))### last term ["BetaTC","DIP"]

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


if __name__ == '__main__':
    # Go!
    #start_time = time.time()
    aicrowd_helpers.execution_start()
    aicrowd_helpers.register_progress(0.)
    # Training loop
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train(epoch)
    # Almost done...
    elapsed_time = time.time() - start_time
    if args.pruning:
        file_name = "p_model.pth"
    else:
        file_name = "model.pth"
    torch.save(model.state_dict(),file_name)
    aicrowd_helpers.register_progress(0.90)
    # Export the representation extractor
    pyu.export_model(RepresentationExtractor(model.encoder.cuda(), 'mean').to(device),
                     input_shape=(1, 3, 64, 64), model_name=args.model)
    # Done!
    aicrowd_helpers.register_progress(1.0)
    aicrowd_helpers.submit()
