import torch
import torch.nn as nn
from torchvision import models, transforms


class Network(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        # Initializng the network
        vgg16 = models.vgg16(pretrained = True)
        
        # Freezing training for all layers
        for param in vgg16.features.parameters():
            param.require_grad = False
        
        # Replacing the last layer of classifier with new layer
        vgg16.classifier[6] = nn.Linear(vgg16.classifier[6].in_features, 20) 
        
        # Adding more layers
        self.vgg16 = vgg16
        self.hidden1 = nn.Linear(20,16)
        self.output = nn.Linear(16,8)
        self.drpout = nn.Dropout(0.2)
                                
    def forward(self, x):
        
        # Implement the forward pass
        x = self.vgg16(x)
        x = self.drpout(x)
        x = self.hidden1(x)
        x = self.output(x)
        
        return x 
    
    def save_model(self):
    
        # Saving the model's weitghts
        torch.save(self.state_dict(), 'model')

