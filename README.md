# kaggle-challange-imageclassification

## overview
This is a kaggle challange to develop a custom dataset and build a classification model. Given are the images segregated by class in folders, with the folder name being the class name. 

## dataset implementaion
*data.py*  
The custom dataset module is implemented in this file. glob module is used to fetch the path of an image and also to identify the class name and index. the image is fetched using the image.open function (pil module).

## model
*model.py*  
Using transfer learning a VGG16 model pre-trianed on Imagenet dataset is imported from pytorch models library. the final layer of the VGG16 classifier is modefied to have 20 neurons. an additional MLP head is added to reduce the output to 8 classes.

