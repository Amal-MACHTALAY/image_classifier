"""
@author: amal machtalay
"""

import torch 
from torch import optim
from torchvision import models

def save_checkpoint(model, optimizer, name, epoch, image_datasets, path):
    model.class_to_idx = image_datasets['training'].class_to_idx
    torch.save({'name': name,
                'epoch': epoch,
                'class_to_idx': model.class_to_idx,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'classifier': model.classifier,
                }, path)
    return 0


# function that loads a checkpoint and rebuilds the model
def rebuild_model(path):
    loaded_checkpoint = torch.load(path)
    loaded_name = loaded_checkpoint['name']
    loaded_epoch = loaded_checkpoint['epoch']
    ### Remember to first initialize the model and optimizer, then load the dictionary
    model = models.vgg16(pretrained=True)
    model.class_to_idx = loaded_checkpoint['class_to_idx']
    model.load_state_dict = loaded_checkpoint['model_state_dict']
    model.classifier = loaded_checkpoint['classifier']
    optimizer = optim.SGD(model.classifier.parameters(), lr=0.001)
    optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
    
    return loaded_name, model, optimizer, loaded_epoch
