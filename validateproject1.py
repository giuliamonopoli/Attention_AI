
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
import timm
import litdata
from time import process_time



# Get your fine-tuned models here.
def load_my_models():
    """
    Loads the fine-tuned models (full and LoRA).
    These should be saved in the output folder.
    Return the models in the order indicated below,
    so the teachers can test them.
    """
    full_model = torch.load('output/Full_best_model_woof__acc_78.6357__lr_0.0001__epoch_5__2023-09-08_21-53-06.pth')
    lora_model = torch.load('output/lora_model.pth')
    return full_model, lora_model


def test_load_my_models():
    full_model, lora_model = load_my_models()
    # full_model = load_my_models()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_model = full_model.to(device)
    lora_model = lora_model.to(device)

    full_model.eval()
    lora_model.eval()

    # Send an example through the models, to check that they loaded properly
    test_img = torch.load('../output/test_img.pth')
    with torch.no_grad():
        _ = full_model(test_img.unsqueeze(0).to(device))
        _ = lora_model(test_img.unsqueeze(0).to(device))

    print("Done!")
if __name__ == '__main__':
    test_load_my_models()
