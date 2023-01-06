"""
@author: amal machtalay
"""

import torch 
import numpy as np
from PIL import Image

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image)
    # Resize the images where the shortest side is 256 pixels, keeping the aspect ratio
    aspect_ratio = pil_image.width / pil_image.height

    if aspect_ratio > 1:
        pil_image.thumbnail((round(aspect_ratio * 256), 256))
    else:
        pil_image.thumbnail((256, round(256 / aspect_ratio)))
    
    # Crop out the center 224x224 portion of the image
    left = (pil_image.width - 224) / 2
    right = (pil_image.width + 224) / 2
    top = (pil_image.height - 224) / 2
    bottom = (pil_image.height + 224) / 2  
    pil_image = pil_image.crop((round(left), round(top), round(right), round(bottom)))
    
    # Convert 0-255 values to 0-1 values
    np_image = np.array(pil_image)/ 255
    
    # Standardization
    means = np.array([0.485, 0.456, 0.406])
    stdv = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - means) / stdv
    
    # Move color channels to first dimension as expected by PyTorch
    np_image = np_image.transpose(2, 0, 1)
    
    return np_image


def predict_fct(image_path, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to(device)
    model.eval()
    
    # TODO: Implement the code to predict the class from an image file
    np_image = process_image(image_path) # Process image
    
    # numpy array to PyTorch tensor
    tf_image = torch.from_numpy(np_image).type(torch.FloatTensor).unsqueeze(0)
    tf_image = tf_image.to(device) # Move to the GPU

    output = model.forward(tf_image)
    ps = torch.exp(output) # get the class probabilities from log-softmax
    top_ps, top_idx = ps.topk(topk)
    
    probs = [float(p) for p in top_ps[0]]
    idx_to_class = {model.class_to_idx[j]: j for j in model.class_to_idx}
    classes = [idx_to_class[int(i)] for i in top_idx[0]]
        
    return probs, classes
