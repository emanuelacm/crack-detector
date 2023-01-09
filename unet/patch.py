import os
import numpy as np
import cv2
import torch
from patchify import patchify, unpatchify

from model import build_unet


if __name__ == "__main__":
    """ Video Path """
    video_path = "videos/20220730_170735.mp4" #"/home/emanuel/tesis/cod_1/videos/test7.mp4"

    """ Hyperparameters """
    H_i = 448
    W_i = 448
    size = (W_i, H_i)
    checkpoint_path = "/home/emanuel/tesis/cod_1/saved_models/checkpoint.pth"

    """ Load the model """

    device = torch.device('cuda')
    model = build_unet()
 
    model = model.to(device)

    model.load_state_dict(torch.load(checkpoint_path,map_location = device))
    model.eval()

    """ Reading frames """
    vs = cv2.VideoCapture(video_path)
    _, frame = vs.read()
    H, W, _ = frame.shape
    vs.release()

    H_v = int(H / 2)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('video_output.mp4',fourcc, 15.0,(W,H_v))

    cap = cv2.VideoCapture(video_path)
    
    idx = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            cap.release()
            out.release()
            break
        H, W, _ = frame.shape
        ori_frame = frame
        print(ori_frame.shape)
        frame = cv2.resize(frame, (448*3,448*5)) 
        print(frame.shape)
        
        patches = patchify(frame,(H_i,W_i,3), step = 448)
        print('El tamano de patches es:' + str(patches.shape))
        
        for patch in patches:
            print('El tamano de patch es:' + str(patch.shape))
            
            pred_0 = patch[1,0,:,:,:]
            pred_1 = patch[1,0,:,:,:]
            pred_2 = patch[2,0,:,:,:]
            
            if (pred_0 == pred_1).all:
                print('pred0 y pred1 son iguales')
            
            if (pred_1 == pred_2).all:
                print('pred1 y pred2 son iguales')
                
            part = patch[:,:,0,0,0]
            print(part.shape)
            
            comp = np.ndarray((3,1,448,448,3))
            print(comp.shape)
            
            comp[:,:,0,0,0] = part
            comp[1,0,:,:,:] = pred_0 
            
            if (comp == patch).all:
                print('comp y patch son iguales')
       
       
        print('El tamano de patches es:' + str(patches.shape))
        frame_comp = unpatchify(patches,(448*5,448*3,3))
        print('El tamano de frame_comp es:' + str(frame_comp.shape))
        frame_comp = cv2.resize(frame_comp, (W, H))
        print('El tamano de frame_comp es:' + str(frame_comp.shape))   
        
        
        predicted_patches = []
for i in range(patches.shape[0]):
    for j in range(patches.shape[1]):
        print(i,j)
        
        single_patch = patches[i,j,:,:]
        single_patch_norm = np.expand_dims(normalize(np.array(single_patch), axis=1),2)
        single_patch_input=np.expand_dims(single_patch_norm, 0)

#Predict and threshold for values above 0.5 probability
        single_patch_prediction = (model.predict(single_patch_input)[0,:,:,0] > 0.5).astype(np.uint8)
        predicted_patches.append(single_patch_prediction)

predicted_patches = np.array(predicted_patches)

predicted_patches_reshaped = np.reshape(predicted_patches, (patches.shape[0], patches.shape[1], 256,256) )
reconstructed_image = unpatchify(predicted_patches_reshaped, large_image.shape) 