# Guide to perform texture generation with multi-resolution  

In this directory you can do multiple things : 
   
    1. Train the neural network on a new texture
    2. Use pre-existing weights and render textures

## 1. Train the nerual network on a new texture

    1. Open the sshInfinteRes.py file
    2. Go to the end of the file
    3. Fill the requested parameters : 
        - path to the texture on which you want to train the network
        - the highest resolution of the image you want. Try first (128,128) then try to upgrade
        - the number of resolution you want the network to train on. You need to enter a list of >=1 scale factor. Three resolution is ideal but you can try more.
        - the number of steps for the training. We advise 1000 to 3000 steps.
    4. In the directory ./trainmodel you can find a folder containing the weights of the neural network you just trainned.

## 2. Use pre-existing weights and render textures

    1. Open loadweightsInfiniteRes.py
    2. Go to the end of the file
    3. Fill the requested parameters : 
        - path to the weights. DO NOT CHANGE THE NAME
        - The size of the image you want to generate. Note that if you trainned on (256,256) rendering a image of higher size do not increase the resolution. It just increase the general size.
        - If you want to be able to compare with the original image you can enter the path to the original image
        - steps of rendering. We advise 200-800.


