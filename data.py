import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
from PIL import Image
import glob
from natsort import natsorted
from torchvision import transforms
class ChristmasImages(Dataset):
    
    def __init__(self, path, training=True):
        super().__init__()
        self.training = training
        self.transform = transforms.Compose(            
            [transforms.ToTensor(),
            transforms.Resize((128,128)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        # If training == True, path contains subfolders
        # containing images of the corresponding classes
        if(self.training):
            self.path = path 
            self.images_path = natsorted(glob.glob(self.path + "/**/*.png"))
            self.labels = []
            for image_path in self.images_path:
                directory = os.path.split(os.path.split(image_path)[0])[1]
                self.labels.append(os.listdir(path).index(directory))
                      
            
        # If training == False, path directly contains
        # the test images for testing the classifier
        else:
            self.path = path
            self.images_path = natsorted(glob.glob(self.path + "/*.png"))
            

        
        
    def __getitem__(self, index):
        # If self.training == False, output (image, )
        # where image will be used as input for your model
        if(self.training):
            image = self.transform(Image.open(self.images_path[index]).convert('RGB')) #opening the image, converting it into rgb and transforming
            label = self.labels[index]
            return (image, label) 
        else:
            image = self.transform(Image.open(self.images_path[index]).convert('RGB')) #opening the image, converting it into rgb and transforming
            
            return (image, )                          
    def __len__(self):
        return len(self.images_path) 
