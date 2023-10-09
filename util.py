import os
import torch
import numpy as np
import json
import glob
import soundfile as sf
import shutil
import yaml

def load_index(data_dir, ext=['wav','mp3'], max_len=10000, inplace=False):
    dataset = {}

    print(f"=>Loading indices from {data_dir}")
    if inplace:
        json_path = os.path.join(data_dir, data_dir.split('/')[-1] + ".json")
    else:
        json_path = os.path.join('data', data_dir.split('/')[-1] + ".json")
    if not os.path.exists(json_path):
        idx = 0
        for fpath in glob.iglob(os.path.join(data_dir,'**/*.*'), recursive=True):
            if len(dataset) >= max_len:
                break
            if fpath.split('.')[-1] in ext: 
                dataset[str(idx)] = fpath
                idx += 1

        with open(json_path, 'w') as fp:
            json.dump(dataset, fp)
    
    else:
        print(f"Index exists. Loading indices from {json_path}")
        with open(json_path, 'r') as fp:
            dataset = json.load(fp)

    assert len(dataset) > 0
    return dataset

def load_augmentation_index(data_dir, splits, ext=['wav','mp3'], shuffle_dataset=True):
    dataset = {'train' : [], 'test' : [], 'validate': []}
    json_path = os.path.join(data_dir, data_dir.split('/')[-1] + ".json")
    if not os.path.exists(json_path):
        fpaths = glob.glob(os.path.join(data_dir,'**/*.*'), recursive=True)
        fpaths = [p for p in fpaths if p.split('.')[-1] in ext]
        dataset_size = len(fpaths)   
        indices = list(range(dataset_size))
        if shuffle_dataset :
            np.random.seed(42)
            np.random.shuffle(indices)
        if type(splits) == list or type(splits) == np.ndarray:
            splits = [int(splits[ix]*dataset_size) for ix in range(len(splits))]
            train_idxs, valid_idxs, test_idxs = indices[:splits[0]], indices[splits[0]: splits[0] + splits[1]], indices[splits[1]:]
            dataset['validate'] = [fpaths[ix] for ix in valid_idxs]
        else:
            splits = int(splits*dataset_size)
            train_idxs, test_idxs = indices[:splits], indices[splits:]
        
        dataset['train'] = [fpaths[ix] for ix in train_idxs]
        dataset['test'] = [fpaths[ix] for ix in test_idxs]

        with open(json_path, 'w') as fp:
            json.dump(dataset, fp)
    
    else:
        with open(json_path, 'r') as fp:
            dataset = json.load(fp)

    return dataset


def get_frames(y, frame_length, hop_length):
    # frames = librosa.util.frame(y.numpy(), frame_length, hop_length, axis=0)
    frames = y.unfold(0, size=frame_length, step=hop_length)
    return frames

def qtile_normalize(y, q, eps=1e-8):
    return y / (eps + torch.quantile(y,q=q))


def load_ckp(checkpoint_fpath, model, optimizer, scheduler):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    return model, optimizer, scheduler, checkpoint['epoch'], checkpoint['loss'], checkpoint['valid_acc']

def save_ckp(state,model_name,model_folder,text):
    if not os.path.exists(model_folder): 
        print("Creating checkpoint directory...")
        os.mkdir(model_folder)
    torch.save(state, "{}/model_{}_{}.pth".format(model_folder, model_name, text))

def load_config(config_path):
    with open(config_path, 'r') as fp:
        config = yaml.safe_load(fp)
    return config

def override(config_val, arg):
    return arg if arg is not None else config_val 

def create_fp_dir(resume=None, ckp=None, epoch=1):
    parent_dir = 'logs/emb'
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    if resume is not None:
        ckp_name = resume.split('/')[-1].split('.pt')[0]
    else:
        ckp_name = f'model_{ckp}_epoch_{epoch}'
    fp_dir = os.path.join(parent_dir, ckp_name)
    if not os.path.exists(fp_dir):
        os.mkdir(fp_dir)
    return fp_dir



def create_train_set(data_dir, dest, size=10000):
    if not os.path.exists(dest):
        os.mkdir(dest)
        print(data_dir)
        print(dest)
    for ix,fname in enumerate(os.listdir(data_dir)):
        fpath = os.path.join(data_dir, fname)
        if ix <= size and fpath.endswith('mp3'):
            shutil.move(fpath,dest)
            print(ix)
        if len(os.listdir(dest)) >= size:
            return dest
    
    return dest

def create_downstream_set(data_dir, size=5000):
    src = os.path.join(data_dir, f'fma_downstream')
    dest = data_dir
    # if not os.path.exists(dest):
    #     os.mkdir(dest)   
    # if len(os.listdir(dest)) >= size:
    #     return dest
    for ix,fname in enumerate(os.listdir(src)):
        fpath = os.path.join(src, fname)
        if not fpath.endswith('mp3'):
            continue
        # if ix < size:
        if len(os.listdir(src)) > 500:
            shutil.move(fpath,dest)

    return dest

def preprocess_aug_set_sr(data_dir, sr=22050):
    for fpath in glob.iglob(os.path.join(data_dir,'**/*.wav'), recursive=True):
        y, sr = sf.read(fpath)
        print(sr)
        break
        # sf.write(fpath, data=y, samplerate=sr)
    return



def main():
    # delete checkpoints for epochs under 100
    for fpath in glob.iglob(os.path.join('checkpoint','*.pth'), recursive=True):
        epoch = int(fpath.split('/')[-1].split('_')[-1].split('.')[0])
        if epoch <= 100:
            os.remove(fpath)


if __name__ == '__main__':
    main()