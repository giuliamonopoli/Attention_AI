import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import numpy as np

import torch
import timm
import argparse
import matplotlib.pyplot as plt
import litdata
from torch import nn
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm
import matplotlib.pyplot as plt
import numpy as np
import litdata
import torchvision.transforms as T
import torchvision.transforms.functional as F

class ToRGBTensor:
    
    def __call__(self, img):
        return F.to_tensor(img).expand(3, -1, -1) # Expand to 3 channels
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
#Loads in data and returns the dataloader

# Set seed
seed = 1
batch_size =  128
# Define mean and std from ImageNet data
in_mean = [0.485, 0.456, 0.406]
in_std = [0.229, 0.224, 0.225]
datapath = '/projects/ec232/data/'

# Define postprocessing / transform of data modalities
postprocess = (
    T.Compose([                        # Handles processing of the .jpg image
    ToRGBTensor(), 
    T.Resize((224,224), antialias=None),# Convert from PIL image to RGB torch.Tensor.
    T.Normalize(in_mean, in_std),  # Normalize image to correct mean/std.
]),
nn.Identity(), 
)

# Load training and validation data
traindata = litdata.LITDataset('ImageWoof', datapath).map_tuple(*postprocess)
valdata = litdata.LITDataset('ImageWoof', datapath, train=False).map_tuple(*postprocess)

# Make and return the dataloaders
train_dataloader = DataLoader(traindata, shuffle=True, batch_size=batch_size)
val_dataloader = DataLoader(valdata, shuffle=False, batch_size=batch_size)



dataiter = iter(train_dataloader)
images, labels = next(dataiter)

image = images[0]


 
 
class MyCustomAttentionMap(nn.Module):
    def __init__(self, attention_module):
        super().__init__()
      
        self.num_heads = attention_module.num_heads
        self.head_dim = attention_module.head_dim
        self.scale = attention_module.scale
        self.qkv = attention_module.qkv
        self.q_norm = attention_module.q_norm
        self.k_norm = attention_module.k_norm
        self.attn_drop = attention_module.attn_drop
        self.proj = attention_module.proj
        self.proj_drop = attention_module.proj_drop
        self.attn_matrix = None 

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        self.attn_matrix = attn

        return x





def visualize_attention( img, patch_size, attentions):
    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - \
        img.shape[2] % patch_size
    img = img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size

    

    nh = attentions.shape[1]  # number of head

   
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(
        0), scale_factor=patch_size, mode="nearest")[0].detach().numpy()

    return attentions


def plot_attention( attention):
    n_heads = attention.shape[0]

    plt.figure(figsize=(10, 10))
    image_numpy = image.permute(1, 2, 0).numpy()

        # Plot the image
    plt.imshow(image_numpy)
    plt.imshow(np.mean(attention,0), cmap='viridis',alpha = 0.7)
    # for i in range(n_heads):
    #     plt.subplot(n_heads//3, 3, i+1)
    #     image_numpy = image.permute(1, 2, 0).numpy()

    #     # Plot the image
    #     plt.imshow(image_numpy)
        
    #     plt.imshow(attention[i], cmap='inferno',alpha=0.6)
    #     plt.title(f"Head n: {i+1}")
    plt.tight_layout()
    plt.savefig('attention_plot.png')

    plt.show()



# Define command-line arguments to specify the model type (lora=True or lora=False)
parser = argparse.ArgumentParser(description="Load a ViT model with or without LoRA.")
parser.add_argument("--lora", action="store_true", help="Load the model with LoRA")
args = parser.parse_args()


if args.lora:
    # Load the model with LoRA (model_lora.pth)
    model_path = "lora_model.pth"
else:
    # Load the model without LoRA (full_model.pth)
    model_path = "full_model.pth"

# Load the selected model
device =  "cuda" if torch.cuda.is_available() else "cpu"
modelpath = "lora_model.pth"
model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=10).to(device)
checkpoint = torch.load(modelpath)  # Replace with the path to your saved model checkpoint
model.load_state_dict(checkpoint['model_state_dict'],map_location=torch.device('cpu'))


source_folder = 'source'


model.blocks[-1].attn = MyCustomAttentionMap(model.blocks[-1].attn)

checkpoint_path = model_path  # Replace with the actual path to your saved model file
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])


model.eval()
with torch.no_grad():
    output = model(image.unsqueeze(0))  # Batch size of 1, as it's a single image



attention_matrix = model.blocks[-1].attn.attn_matrix
att = visualize_attention(image, 16, attention_matrix)
plot_attention(att)

