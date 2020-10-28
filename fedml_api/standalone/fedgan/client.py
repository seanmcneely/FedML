import logging 
import torch 

from torch import nn
import torch.autograd as autograd
from fedml_api.model.cv.cnn_wgan_gp import calc_gradient_penalty


class Client:
    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, args, device):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        logging.info("self.local_sample_number = " + str(self.local_sample_number))

        self.discrim_iters = args.discrim_iters
        self.args = args
        self.device = device

        self.criterion = nn.BCELoss().to(device)

        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1
        self.one = one.to(device)
        self.mone = mone.to(device)


    def update_local_dataset(self, client_idx, local_training_data, local_test_data, local_sample_number):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

    def get_sample_number(self):
        return self.local_sample_number

    def train(self, netD, netG):
        for p in netD.parameters():
            p.requires_grad = True

        optimizerD = torch.optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))

        epoch_loss = []
        for iter_d in range(self.discrim_iters):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(self.local_training_data):
                real = x.to(self.device)
                netD.zero_grad()

                D_real = netD(real)
                D_real = D_real.mean()
                D_real.backward(self.mone)
                
                batch_size = len(x)
                noise = torch.randn(batch_size, 128)
                fake = netG(noise).data
                inputv = fake

                D_fake = netD(inputv)
                D_fake = D_fake.mean()
                D_fake.backward(self.one)

                gradient_penalty = calc_gradient_penalty(netD, real.data, fake.data, self.device, self.args._lambda)
                gradient_penalty.backward()

                optimizerD.step()

                D_cost = D_fake - D_real + gradient_penalty
                batch_loss.append(D_cost)
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return netD.cpu().state_dict(), sum(epoch_loss) / len(epoch_loss)