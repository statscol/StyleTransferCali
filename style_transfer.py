
##code based on udacity free course intro to deep learning with pytorch https://www.udacity.com/course/deep-learning-pytorch--ud188


from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import requests
from torchvision import transforms, models
import argparse
from tqdm import tqdm

MODEL = models.vgg19(pretrained=True).features
CONV_WEIGHTS = {'conv1_1': 1.,
                 'conv2_1': 0.8,
                 'conv3_1': 0.5,
                 'conv4_1': 0.3,
                 'conv5_1': 0.1}

##High value for style weight will generate more similar-to-style results 
CONTENT_WEIGHT = 1  
STYLE_WEIGHT = 1e6  


##using VGG as backbone
for param in MODEL.parameters():
    param.requires_grad_(False)

#check if cuda is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL.to(device)


def load_image(img_path, max_size=500, shape=None):
    ''' Load in and transform an image, making sure the image
       is <= 500 pixels in the x-y dims.'''
    
    image = Image.open(img_path).convert('RGB')
    
    # large images will slow down processing
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    
    if shape is not None:
        size = shape
    
    ##Standardize image pipeline
    in_transform = transforms.Compose([
                        transforms.Resize(size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), 
                                             (0.229, 0.224, 0.225))])

    image = in_transform(image).unsqueeze(0) ##add dimension
    
    return image

def im_convert(tensor):
    """ Display a tensor as an image. """
    
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image

def get_features(image, model, layers=None):
    """ Run an image forward through a model and get the features for 
        a set of layers. Default layers are for VGGNet matching Gatys et al (2016)
    """
    
    ##conv layers to use from MODEL
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1', 
                  '10': 'conv3_1', 
                  '19': 'conv4_1',
                  '21': 'conv4_2',  ## content representation
                  '28': 'conv5_1'}
        
    features = {}
    x = image
    # model._modules is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
            
    return features

def gram_matrix(tensor):
    """ Calculate the Gram Matrix of a given tensor 
    """
    
    batch_size, d, h, w = tensor.size()
    tensor=tensor.view(batch_size * d, h * w)
    
    gram = torch.mm(tensor,tensor.t())
    return gram 


def train(target_img,model,style_weights,content_weight):

    optimizer = optim.Adam([target_img], lr=0.003)
    steps = 3000  # decide how many iterations to update target image 

    for i in tqdm(range(1, steps+1)):
    
        target_features = get_features(target,model)
        content_loss = torch.mean((target_features['conv4_2']-content_features['conv4_2'])**2)
        
        # the style loss
        # initialize the style loss to 0
        style_loss = 0
        # iterate through each style layer and add to the style loss
        for layer in style_weights:
            # get the "target" style representation for the layer
            target_feature = target_features[layer]
            _, d, h, w = target_feature.shape
            
            target_gram = gram_matrix(target_feature)
            style_gram = gram_matrix(style_features[layer])
            layer_style_loss =style_weights[layer] * torch.mean((target_gram-style_gram)**2)
            style_loss += layer_style_loss / (d * h * w)
            
            
        ##total loss
        total_loss = content_weight*content_loss + style_loss * style_loss
        
        ## restart gradient directions after batch     
        optimizer.zero_grad()
        total_loss.backward()
        ## update weights
        optimizer.step()
    print("TOTAL LOSS: {}".format(total_loss.item()))
    return target_img

if __name__=="__main__":

    parser=argparse.ArgumentParser(description="Style Transfer Test")
    parser.add_argument("-i","--input",help="input image filepath",dest="input_image")
    parser.add_argument("-s","--style",help="Style image filepath",dest="style_image")
    parser.add_argument("-o","--output",help="Output image filepath",dest="out_path")
    args=parser.parse_args()

    PATH_TO_IMG=args.input_image
    PATH_TO_IMG_STYLE=args.style_image
    content = load_image(PATH_TO_IMG).to(device)
    # resize to match input image shape
    style = load_image(PATH_TO_IMG_STYLE, shape=content.shape[-2:]).to(device)

    content_features = get_features(content, MODEL)
    style_features = get_features(style, MODEL)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
    #target image will be set as the input image PATH_TO_IMG for initialization
    target = content.clone().requires_grad_(True).to(device)
    print("[STARTING TRAINING]")
    image_out=train(target,MODEL,CONV_WEIGHTS,CONTENT_WEIGHT)

    ##image out
    fig, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(20, 10))
    ax1.imshow(im_convert(content))
    ax1.set_title("Input Image")
    ax2.imshow(im_convert(style))
    ax2.set_title("Style Image")
    ax3.imshow(im_convert(target))
    ax3.set_title("Result")
    fig.savefig("{}/output.jpg".format(args.out_path))