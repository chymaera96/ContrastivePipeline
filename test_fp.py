import os
import random
import numpy as np
import torch
import torch.nn as nn
import argparse
import faiss
import json
import shutil

from util import load_ckp, save_ckp, create_train_set
from sfnet.data import NeuralfpDataset
from sfnet.modules.simclr import SimCLR
from sfnet.modules.residual import SlowFastNetwork, ResidualUnit
from sfnet.gpu_transformations import GPUTransformNeuralfp

from eval import get_index, load_memmap_data, eval_faiss
from torch.utils.data.sampler import SubsetRandomSampler



# Directories
root = os.path.dirname(__file__)

ir_dir = os.path.join(root,'data/augmentation_datasets/ir_filters')
noise_dir = os.path.join(root,'data/augmentation_datasets/noise')

parser = argparse.ArgumentParser(description='Neuralfp Testing')
parser.add_argument('--test_dir', default='', type=str)
parser.add_argument('--fp_dir', default='fingerprints', type=str)
parser.add_argument('--ckp', default='', type=str)
parser.add_argument('--query_lens', default='1 3 5 9 11 19', type=str)
parser.add_argument('--n_dummy_db', default=500, type=int)
parser.add_argument('--n_query_db', default=20, type=int)
parser.add_argument('--compute_fp', default=True, type=bool)
parser.add_argument('--small_test', default=False, type=bool)
parser.add_argument('--nb', default=False, type=bool)


device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

def create_fp_db(dataloader, augment, model, output_root_dir, verbose=True):
    fp_q = []
    fp_db = []
    print("=> Creating query and db fingerprints...")
    for idx, audio in enumerate(dataloader):
        audio = audio.to(device)
        x_i, x_j = augment(audio, audio)
        # x_i = torch.unsqueeze(db[0],1)
        # x_j = torch.unsqueeze(q[0],1)
        with torch.no_grad():
            _, _, z_i, z_j= model(x_i,x_j)        

        print(f'Shape of z_i {z_i.shape} inside the create_fp_db function')
        fp_db.append(z_i.detach().cpu().numpy())
        fp_q.append(z_j.detach().cpu().numpy())

        if verbose and idx % 10 == 0:
            print(f"Step [{idx}/{len(dataloader)}]\t shape: {z_i.shape}")
        # fp = torch.cat(fp)
    
    fp_db = np.concatenate(fp_db)
    fp_q = np.concatenate(fp_q)
    arr_shape = (len(fp_db), z_i.shape[-1])


    arr_q = np.memmap(f'{output_root_dir}/query.mm',
                    dtype='float32',
                    mode='w+',
                    shape=arr_shape)
    
    arr_q[:] = fp_q[:]
    arr_q.flush(); del(arr_q)   #Close memmap

    np.save(f'{output_root_dir}/query_shape.npy', arr_shape)

    arr_db = np.memmap(f'{output_root_dir}/db.mm',
                    dtype='float32',
                    mode='w+',
                    shape=arr_shape)
    arr_db[:] = fp_db[:]
    arr_db.flush(); del(arr_db)   #Close memmap

    np.save(f'{output_root_dir}/db_shape.npy', arr_shape)

def create_dummy_db(dataloader, augment, model, output_root_dir, fname='dummy_db', verbose=True):
    fp = []
    print("=> Creating dummy fingerprints...")
    for idx, audio in enumerate(dataloader):
        audio = audio.to(device)
        x_i, _ = augment(audio, audio)
        # x_i = torch.unsqueeze(db[0],1)
        with torch.no_grad():
            _, _, z_i, _= model(x_i,x_i)  

        print(f'Shape of z_i {z_i.shape} inside the create_dummy_db function')
        fp.append(z_i.detach().cpu().numpy())
        
        if verbose and idx % 10 == 0:
            print(f"Step [{idx}/{len(dataloader)}]\t shape: {z_i.shape}")
        # fp = torch.cat(fp)
    
    fp = np.concatenate(fp)
    arr_shape = (len(fp), z_i.shape[-1])

    arr = np.memmap(f'{output_root_dir}/{fname}.mm',
                    dtype='float32',
                    mode='w+',
                    shape=arr_shape)
    arr[:] = fp[:]
    arr.flush(); del(arr)   #Close memmap

    np.save(f'{output_root_dir}/{fname}_shape.npy', arr_shape)


def main():

    args = parser.parse_args()
    # json_dir = load_index(data_dir)

    # Hyperparameters
    random_seed = 42
    shuffle_dataset =True
            
    print("Loading Model...")
    model = SimCLR(encoder=SlowFastNetwork(ResidualUnit, layers=[1,1,1,1])).to(device)

       
    if os.path.isfile(args.ckp):
        print("=> loading checkpoint '{}'".format(args.ckp))
        checkpoint = torch.load(args.ckp)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(args.ckp))

    if not args.nb:

        print("Creating dataloaders ...")
        dataset = NeuralfpDataset(path=args.test_dir, transform=GPUTransformNeuralfp(ir_dir=ir_dir, noise_dir=noise_dir,sample_rate=22050), train=False)


        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split1 = args.n_dummy_db
        split2 = args.n_query_db
        if shuffle_dataset :
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        dummy_indices, query_db_indices = indices[:split1], indices[split1: split1 + split2]
        print(f"Length of indices {len(dummy_indices)} {len(query_db_indices)}")

        dummy_db_sampler = SubsetRandomSampler(dummy_indices)
        query_db_sampler = SubsetRandomSampler(query_db_indices)

        

        dummy_db_loader = torch.utils.data.DataLoader(dataset, batch_size=1, 
                                                shuffle=False,
                                                num_workers=4, 
                                                pin_memory=True, 
                                                drop_last=False,
                                                sampler=dummy_db_sampler)
        
        query_db_loader = torch.utils.data.DataLoader(dataset, batch_size=1, 
                                                shuffle=False,
                                                num_workers=4, 
                                                pin_memory=True, 
                                                drop_last=False,
                                                sampler=query_db_sampler)


        if not os.path.exists(args.fp_dir):
            os.mkdir(args.fp_dir)

        if args.compute_fp == True:
            create_fp_db(query_db_loader, model, args.fp_dir)
            create_dummy_db(dummy_db_loader, model, args.fp_dir)

        if args.small_test:
            index_type = 'l2'
        else:
            index_type = 'ivfpq'
        eval_faiss(emb_dir=args.fp_dir, test_ids='all', test_seq_len=args.query_lens, index_type=index_type)

    else:

        print("Creating dataloader ...")
        dataset_q = NeuralfpDataset(path='/content/DLAM_coursework/data/query', train=False)
        dataset_db = NeuralfpDataset(path='/content/DLAM_coursework/data/db', train=False)


        query_loader = torch.utils.data.DataLoader(dataset_q, batch_size=1, 
                                                shuffle=False,
                                                num_workers=0, 
                                                pin_memory=True, 
                                                drop_last=False)

        db_loader = torch.utils.data.DataLoader(dataset_db, batch_size=1, 
                                                shuffle=False,
                                                num_workers=0, 
                                                pin_memory=True, 
                                                drop_last=False)

        create_dummy_db(query_loader, model, args.fp_dir, fname='query')
        create_dummy_db(db_loader, model, args.fp_dir, fname='db')

        hit_rates = eval_faiss(emb_dir=args.fp_dir, test_ids='all', test_seq_len='1 4 6', index_type='l2')
        print(f'Top-1 exact hit rate = {hit_rates[0]}')
        print(f'Top-1 near hit rate = {hit_rates[1]}')
        print(f'Top-3 exact hit rate = {hit_rates[2]}')
        print(f'Top-10 exact hit rate = {hit_rates[3]}')



if __name__ == '__main__':
    main()