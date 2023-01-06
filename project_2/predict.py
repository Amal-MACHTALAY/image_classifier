"""
@author: amal machtalay
"""

import torch 
import argparse
from save_load_checkpoint import rebuild_model
from predict_fct import predict_fct
import json


def main():
    
    # Use GPU if available
    # define device as the first visible cuda device 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',type=str,default='flowers/test/37/image_03741.jpg',help='Path to the folder of datas')
    parser.add_argument('--arch',type=str,default='vgg',help='CNN Model Architecture')
    parser.add_argument('--ldir',type=str,default='checkpoint.pth',help='directory to load checkpoints')
    parser.add_argument('--top_k',type=int,default=3,help='top K most likely classes')
    arg = parser.parse_args()
   
    _, model, _, _ = rebuild_model(arg.ldir)

    probs, classes = predict_fct(arg.dir, model, device, topk=arg.top_k)
    # Convert class id into its name
    # Load json file
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        names = []
        # Find flower names corresponding to predicted categories
        for classe in classes:
            names.append(cat_to_name[str(classe)])
            
    print("name: ", names) 
    print("probs: ", probs)
    print("classes: ", classes)

# Call to main function to run the program
if __name__ == "__main__":
    main()
    
    
# python predict.py --dir flowers/test/102/image_08042.jpg --arch vgg --ldir checkpoint.pth --top_k 7 > predict.txt 
    



    


