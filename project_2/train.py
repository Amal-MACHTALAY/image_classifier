"""
@author: amal machtalay
"""


import torch 
from torchvision import models
from torch import nn, optim
import argparse
from train_classifier import Classifier, train_test_Classifier
from transforms import transforms_func
from workspace_utils import active_session
from save_load_checkpoint import save_checkpoint

def main():
    
    # Use GPU if available
    # define device as the first visible cuda device 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',type=str,default='flowers/',help='Path to the folder of datas')
    parser.add_argument('--arch',type=str,default='vgg',help='CNN Model Architecture')
    parser.add_argument('--sdir',type=str,default='checkpoint.pth',help='directory to save checkpoints')
    parser.add_argument('--lr',type=float,default=0.003,help='learning rate')
    parser.add_argument('--epch',type=int,default=20,help='number of epochs')
    arg = parser.parse_args()
    
    if arg.arch=='vgg':
        # Load a pre-trained network 
        model = models.vgg19(pretrained=True)
        
        # freeze feature parameters of my vgg19 model
        for param in model.features.parameters():
            param.requires_grad = False
        
        in_size = 25088
        out_size = 102
        hidden_layers = [4096, 1024]
        dropout = 0.1
        model.classifier = Classifier(in_size, out_size, hidden_layers, dropout)
        # Define a Loss function
        loss_func = nn.NLLLoss()
        # SGD algorithm for Stochastic Optimization    
        optimizer = optim.SGD(model.classifier.parameters(), lr=arg.lr)
        
    image_datasets, dataloaders = transforms_func(arg.dir)
    train_loaders = dataloaders['training']
    valid_loaders = dataloaders['validation']
        
    # Train and test Classifier
    with active_session():
        train_test_Classifier(arg.epch, model, loss_func, optimizer, train_loaders, valid_loaders, device)
    
    # Save the checkpoint
    save_checkpoint(model, optimizer, arg.arch, arg.epch, image_datasets, arg.sdir)

# Call to main function to run the program
if __name__ == "__main__":
    main()
    
    
# python train.py --dir flowers/ --arch vgg --lr 0.002 --epch 50 --sdir checkpoint.pth > train.txt



