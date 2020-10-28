import copy 
import logging 

import numpy as np
import torch
import torch.autograd as autograd
import torch.optim as optim
#import wandb
import imageio 

from fedml_api.standalone.fedgan.client import Client 

class FedGanTrainer(object):
    def __init__(self, dataset, generator_model, discriminator_model, device, args):
        self.device = device 
        self.args = args
        [train_data_num, test_data_num, train_data_global, test_data_global,
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num

        self.generator_global = generator_model
        self.generator_global.train()
        self.discriminator_global = discriminator_model
        self.discriminator_global.train()
        self.optimizerG = optim.Adam(self.generator_global.parameters(), lr=1e-4, betas=(0.5, 0.9))

        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1
        self.one = one.to(device)
        self.mone = mone.to(device)

        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict)


    def setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict):
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_per_round):
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                train_data_local_num_dict[client_idx], self.args, self.device)
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")


    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        num_clients = min(client_num_in_total, client_num_per_round)
        np.random.seed(round_idx)
        client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        return client_indexes

    def generate_image(self, round):
        noise = torch.randn(self.args.batch_size, 128)
        noise = noise.to(self.device)
        with torch.no_grad():
            samples = self.generator_global(noise)
        
        samples = samples.view(self.args.batch_size, 28, 28)
        samples = samples.cpu().data.numpy()

        for idx, sample in enumerate(samples):
            imageio.imwrite('./test/' + str(round) + '_' + str(idx) + '.png', samples[idx])


    def train(self):
        for round_idx in range(self.args.comm_round):
            logging.info("################Communication round : {}".format(round_idx))
            self.train_discriminator(round_idx)
            self.train_generator(round_idx)
            if(round_idx % 500 == 499):
                self.generate_image(round_idx)

    def train_discriminator(self, round_idx):
        client_indexes = self.client_sampling(round_idx, self.args.client_num_in_total, self.args.client_num_per_round)
        logging.info("client_indexes = " + str(client_indexes))

        self.discriminator_global.train()
        w_locals, loss_locals = [], []

        for idx, client in enumerate(self.client_list):
            client_idx = client_indexes[idx]
            client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                            self.test_data_local_dict[client_idx],
                                            self.train_data_local_num_dict[client_idx])
            w, loss = client.train(netD=copy.deepcopy(self.discriminator_global).to(self.device), netG=copy.deepcopy(self.generator_global).to(self.device))
            w_locals.append((client.get_sample_number(), copy.deepcopy(w)))
            loss_locals.append(loss.detach().clone())
            logging.info('Client {:3d}, loss {:.3f}'.format(client_idx, loss))

            w_discriminator_global = self.aggregate_discriminator(w_locals)
            self.discriminator_global.load_state_dict(w_discriminator_global)

            disc_loss_avg = sum(loss_locals) / len(loss_locals)
            logging.info('Round {:3d}, Average discriminator loss {:.3f}'.format(round_idx, disc_loss_avg))


    def train_generator(self, round_idx):
        netD = self.discriminator_global
        netG = self.generator_global

        for p in netD.parameters():
            p.requires_grad = False 
        
        netG.zero_grad()

        noise = torch.randn(self.args.batch_size, 128)
        noise = noise.to(self.device)
        fake = netG(noise)
        G = netD(fake)
        G = G.mean()
        G.backward(self.mone)
        G_cost = -G
        logging.info('Round {:3d}, Global generator loss {:.3f}'.format(round_idx, G_cost))
        self.optimizerG.step()

    def aggregate_discriminator(self, w_locals):
        num_clients = len(w_locals)
        (sample_num, averaged_params) = w_locals[0]

        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                if i == 0:
                    averaged_params[k] = local_model_params[k] / num_clients
                else:
                    averaged_params[k] += local_model_params[k] / num_clients
        return averaged_params















