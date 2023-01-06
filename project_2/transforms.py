"""
@author: amal machtalay
"""

import torch
from torchvision import transforms, datasets

# Define your transforms for the training, validation, and testing sets
def transforms_func(data_dir):
    train_dir = data_dir + 'train'
    valid_dir = data_dir + 'valid'
    test_dir = data_dir + 'test'
    
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                           transforms.RandomRotation(45),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomVerticalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                std=[0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                               std=[0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                               std=[0.229, 0.224, 0.225])])

    #data_transforms = {"training": train_transforms, "validation": valid_transforms, "testing": test_transforms}
    # Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)

    image_datasets = {"training": train_datasets, "validation": valid_datasets, "testing": datasets}

    # Using the image datasets and the trainforms, define the dataloaders
    train_loaders = torch.utils.data.DataLoader(train_datasets, batch_size=50, shuffle=True)
    valid_loaders = torch.utils.data.DataLoader(valid_datasets, batch_size=30)
    test_loaders = torch.utils.data.DataLoader(test_datasets, batch_size=30)

    dataloaders = {"training": train_loaders, "validation": valid_loaders, "testing": test_loaders}
    
    return image_datasets, dataloaders
