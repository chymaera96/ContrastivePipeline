# ContrastivePipeline
A simple contrastive learning backbone for audio fingerprinting. This repository is currently in development. The baseline framework is a PyTorch implementation of "NEURAL AUDIO FINGERPRINT FOR HIGH-SPECIFIC AUDIO RETRIEVAL BASED ON CONTRASTIVE LEARNING" (Chang, Lee et.al., 2021). 

## Installation Guide
The pipeline has a `torch` backend and the query-search (during validation and testing) is based on `faiss-gpu`. The data augmentation relies on `torch-audiomentations`.
```
python -m pip install -r requirements.txt
```
## Training the Baseline
At the moment, the pipeline is optimised only for single-GPU training. Validation is run on a toy set of 120 (100 dummy + 20 query) songs every 10 epochs. To run training in the default setting:
```
python baseline_train.py --config=config/baseline.yaml --encoder=baseline --train_dir=PATH/TO/TRAIN/DATA --valid_dir=PATH/TO/VALID/DATA
```
The path to training and validation data can be either overridden by parsed arguments or changed directly in the config file. The training would also require a noise dataset and a room impulse response dataset, both of which can be found in the neural-audio-fp [repo](https://github.com/mimbres/neural-audio-fp/tree/main). Note that the augmentation datasets provided here are sampled at 8 kHz which is suitable for this framework. 

The training loss and validation accuracy (computed every 10 epochs) can be monitored using `tensorboard`.

```
tensorboard --logdir=runs --port=PORT
```
