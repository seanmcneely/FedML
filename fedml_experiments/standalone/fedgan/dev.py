import argparse
import logging
import os
import sys

import numpy as np
import torch
import copy
#import wandb

#from fedml_api.model.linear.lr import LogisticRegression
from fedml_api.model.cv.cnn_wgan_gp import Discriminator 
from fedml_api.model.cv.cnn_wgan_gp import Generator
from fedml_api.data_preprocessing.MNIST.data_loader import load_partition_data_mnist

from fedml_api.standalone.fedgan.fedgan_trainer import FedGanTrainer



def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--dataset', type=str, default='mnist', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--output_dim', type=int, default=784, 
                        help='dimension of generator images')

    parser.add_argument('--model_dim', type=int, default=64,
                        help='DIM used in layer sizes of generator and discriminator')

    parser.add_argument('--_lambda', type=int, default=10,
                        help='lambda parameter for gradient penalty')

    parser.add_argument('--data_dir', type=str, default='./../../../data/cifar10',
                        help='data directory')

    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='SGD with momentum; adam')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--discrim_iters', type=int, default=5, metavar='EP',
                        help='how many discriminator epochs will be trained locally per generator update')

    parser.add_argument('--client_num_in_total', type=int, default=1000, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=4, metavar='NN',
                        help='number of workers')

    parser.add_argument('--comm_round', type=int, default=12000,
                        help='how many round of communications we shoud use')

    parser.add_argument('--frequency_of_the_test', type=int, default=5,
                        help='the frequency of the algorithms')

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')

    args = parser.parse_args()
    return args

def create_model(args, model_dim, output_dim):
    return Discriminator(model_dim), Generator(model_dim, output_dim)

if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    args = add_args(argparse.ArgumentParser(description='FedAvg-standalone'))
    logger.info(args)
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    logger.info(device)

    dataset_name = args.dataset #TODO ignoring for now to get mnist working
    logging.info("load_data. dataset_name = %s" % dataset_name)
    client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
    train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
    class_num = load_partition_data_mnist(args.batch_size)

    args.client_num_in_total = client_num
    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]

    discriminator_model, generator_model = create_model(args, model_dim=args.model_dim, output_dim=args.output_dim) #TODO create model that uses dims of various datasets
    trainer = FedGanTrainer(dataset, generator_model, discriminator_model, device, args)
    trainer.train()