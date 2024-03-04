import torch 
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from baseline.encoder import Encoder

class EncoderFeatureExtractor(nn.Module):
    def __init__(self, encoder, target_layer):
        super(EncoderFeatureExtractor, self).__init__()
        self.encoder = encoder
        self.target_layer = target_layer
        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.encoder.conv_layers(x)
        x = x.reshape(x.shape[0],-1)
        return x

    def get_gradients(self):
        return self.gradients

    def get_activations(self, x):
        return self.encoder.conv_layers(x)
    
    def get_target_layer(self):
        return self.target_layer
    
    def get_encoder(self):
        return self.encoder
    

class GradCam:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.model.eval()
        self.feature_extractor = EncoderFeatureExtractor(self.model, self.target_layer)
        self.model.to("cuda")
        self.feature_extractor.to("cuda")
        self.gradcam = GradCAM(model=self.feature_extractor, target_layer=self.target_layer)
        
    def generate(self, input_tensor, target_category=None):
        input_tensor = input_tensor.to("cuda")
        target_category = None
        self.model.zero_grad()
        self.feature_extractor.zero_grad()
        self.gradcam(input_tensor, target_category)
        return self.gradcam

    def visualize(self, cam, img):
        heatmap, result = show_cam_on_image(img, cam, use_rgb=True)
        return heatmap, result

class SimilarityToConceptTarget:
    def __init__(self, features):
        self.features = features
    
    def __call__(self, model_output):
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        return cos(model_output, self.features)
    


