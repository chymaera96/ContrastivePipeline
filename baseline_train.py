import os
import numpy as np
import argparse
import torch
import gc
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter


from util import *
from ntxent import ntxent_loss
from sfnet.gpu_transformations import GPUTransformNeuralfp
from sfnet.data_sans_transforms import NeuralfpDataset
from sfnet.modules.simclr import SimCLR
from sfnet.modules.residual import SlowFastNetwork, ResidualUnit
from baseline.encoder import Encoder
from baseline.neuralfp import Neuralfp
from eval import eval_faiss
from test_fp import create_fp_db, create_dummy_db

# Directories
root = os.path.dirname(__file__)
model_folder = os.path.join(root,"checkpoint")


device = torch.device("cuda")


parser = argparse.ArgumentParser(description='Neuralfp Training')
parser.add_argument('--config', default=None, type=str,
                    help='Path to training data')
parser.add_argument('--train_dir', default=None, type=str, metavar='PATH',
                    help='path to training data')
parser.add_argument('--val_dir', default=None, type=str, metavar='PATH',
                    help='path to validation data')
parser.add_argument('--epochs', default=None, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--ckp', default='test', type=str,
                    help='checkpoint_name')
parser.add_argument('--encoder', default='sfnet', type=str)
parser.add_argument('--n_dummy_db', default=None, type=int)
parser.add_argument('--n_query_db', default=None, type=int)



def train(cfg, train_loader, model, optimizer, ir_idx, noise_idx, augment=None):
    model.train()
    loss_epoch = 0
    # return loss_epoch
    # if augment is None:
    #     augment = GPUTransformNeuralfp(ir_dir=ir_idx, noise_dir=noise_idx, sample_rate=sr).to(device)

    for idx, (x_i, x_j) in enumerate(train_loader):

        optimizer.zero_grad()
        x_i = x_i.to(device)
        x_j = x_j.to(device)
        with torch.no_grad():
            x_i, x_j = augment(x_i, x_j)
        h_i, h_j, z_i, z_j = model(x_i, x_j)
        loss = ntxent_loss(z_i, z_j, cfg)
        loss.backward()
        optimizer.step()

        if idx % 10 == 0:
            print(f"Step [{idx}/{len(train_loader)}]\t Loss: {loss.item()}")

        loss_epoch += loss.item()

    return loss_epoch

def validate(epoch, query_loader, dummy_loader, augment, model, output_root_dir):
    model.eval()
    if epoch==1 or epoch % 10 == 0:
        create_dummy_db(dummy_loader, augment=augment, model=model, output_root_dir=output_root_dir, verbose=False)
        create_fp_db(query_loader, augment=augment, model=model, output_root_dir=output_root_dir, verbose=False)
        hit_rates = eval_faiss(emb_dir=output_root_dir, test_ids='all', index_type='l2', n_centroids=64)
        print("-------Validation hit-rates-------")
        print(f'Top-1 exact hit rate = {hit_rates[0]}')
        print(f'Top-1 near hit rate = {hit_rates[1]}')
    else:
        hit_rates = None
    return hit_rates

def main():
    args = parser.parse_args()
    cfg = load_config(args.config)
    writer = SummaryWriter(f'runs/{args.ckp}')
    train_dir = override(cfg['train_dir'], args.train_dir)
    valid_dir = override(cfg['val_dir'], args.val_dir)
    ir_dir = cfg['ir_dir']
    noise_dir = cfg['noise_dir']
    
    # Hyperparameters
    batch_size = cfg['bsz_train']
    learning_rate = cfg['lr']
    num_epochs = override(cfg['n_epochs'], args.epochs)
    model_name = args.ckp
    random_seed = args.seed
    shuffle_dataset = True

    # print(f"Size of train index file {len(load_index(train_dir))}")
    # print(f"Size of validation index file {len(load_index(valid_dir))}")


    # assert data_dir == os.path.join(root,"data/fma_8000")

    print("Intializing augmentation pipeline...")
    noise_train_idx = load_augmentation_index(noise_dir, splits=0.8)["train"]
    ir_train_idx = load_augmentation_index(ir_dir, splits=0.8)["train"]
    noise_val_idx = load_augmentation_index(noise_dir, splits=0.8)["test"]
    ir_val_idx = load_augmentation_index(ir_dir, splits=0.8)["test"]
    gpu_augment = GPUTransformNeuralfp(cfg=cfg, ir_dir=ir_train_idx, noise_dir=noise_train_idx, train=True).to(device)
    cpu_augment = GPUTransformNeuralfp(cfg=cfg, ir_dir=ir_train_idx, noise_dir=noise_train_idx, cpu=True)
    val_augment = GPUTransformNeuralfp(cfg=cfg, ir_dir=ir_val_idx, noise_dir=noise_val_idx, train=False).to(device)

    print("Loading dataset...")
    train_dataset = NeuralfpDataset(cfg=cfg, path=train_dir, train=True, transform=cpu_augment)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True)
    
    valid_dataset = NeuralfpDataset(cfg=cfg, path=valid_dir, train=False)
    print("Creating validation dataloaders...")
    dataset_size = len(valid_dataset)
    indices = list(range(dataset_size))
    split1 = override(cfg['n_dummy'],args.n_dummy_db)
    split2 = override(cfg['n_query'],args.n_query_db)
    
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    dummy_indices, query_db_indices = indices[:split1], indices[split1: split1 + split2]

    dummy_db_sampler = SubsetRandomSampler(dummy_indices)
    query_db_sampler = SubsetRandomSampler(query_db_indices)

    dummy_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, 
                                            shuffle=False,
                                            num_workers=1, 
                                            pin_memory=True, 
                                            drop_last=False,
                                            sampler=dummy_db_sampler)
    
    query_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, 
                                            shuffle=False,
                                            num_workers=1, 
                                            pin_memory=True, 
                                            drop_last=False,
                                            sampler=query_db_sampler)

    
    print("Creating new model...")
    if args.encoder == 'baseline':
        model = Neuralfp(encoder=Encoder()).to(device)
    elif args.encoder == 'sfnet':
        model = SimCLR(encoder=SlowFastNetwork(ResidualUnit, cfg)).to(device)
    # pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Number of trainable parameters: {pytorch_total_params}")
    print(count_parameters(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = cfg['T_max'], eta_min = 1e-7)
       
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            model, optimizer, scheduler, start_epoch, loss_log, hit_rate_log = load_ckp(args.resume, model, optimizer, scheduler)
            output_root_dir = create_fp_dir(resume=args.resume)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            
    else:
        start_epoch = 0
        loss_log = []
        hit_rate_log = []
        output_root_dir = create_fp_dir(ckp=args.ckp, epoch=1)


    print("Calculating initial loss ...")
    best_loss = float('inf')
    best_hr = 0
    # training

    for epoch in range(start_epoch+1, num_epochs+1):
        print("#######Epoch {}#######".format(epoch))
        loss_epoch = train(cfg, train_loader, model, optimizer, ir_train_idx, noise_train_idx, gpu_augment)
        writer.add_scalar("Loss/train", loss_epoch, epoch)
        loss_log.append(loss_epoch)
        output_root_dir = create_fp_dir(ckp=args.ckp, epoch=epoch)
        hit_rates = validate(epoch, query_loader, dummy_loader, val_augment, model, output_root_dir)
        hit_rate_log.append(hit_rates[0] if hit_rates is not None else hit_rate_log[-1])
        if hit_rates is not None:
            writer.add_scalar("Exact Hit_rate (2 sec)", hit_rates[0][0], epoch)
            writer.add_scalar("Exact Hit_rate (4 sec)", hit_rates[0][1], epoch)
            writer.add_scalar("Near Hit_rate (2 sec)", hit_rates[1][0], epoch)

        checkpoint = {
            'epoch': epoch,
            'loss': loss_log,
            'valid_acc' : hit_rate_log,
            'hit_rate': hit_rates,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        save_ckp(checkpoint, model_name, model_folder, 'current')
        if loss_epoch < best_loss:
            best_loss = loss_epoch
            save_ckp(checkpoint, model_name, model_folder, 'best')

        # elif hit_rates is not None and hit_rates[0][0] > best_hr:
        #     best_hr = hit_rates[0][0]
        #     checkpoint = {
        #         'epoch': epoch,
        #         'loss': loss_log,
        #         'valid_acc' : hit_rate_log,
        #         'hit_rate': hit_rates,
        #         'state_dict': model.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'scheduler': scheduler.state_dict()
        #     }
        #     save_ckp(checkpoint,epoch, model_name, model_folder)
            
        scheduler.step()
    
  
        
if __name__ == '__main__':
    main()