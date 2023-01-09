from PIL import Image
import os
import os.path
import pathlib
import numpy as np
import torch
import cv2
import math

from model import build_unet

from pathlib import Path

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from skimage.measure import label, regionprops_table


carpetas = ['grietas_longitudinales','grietas_transversales','grietas_cocodrilo']

pixel_h = 0.0006348867037195382
pixel_w = 0.0006241547903880162

def delete_excess(root):

    dirlist = [ item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item)) ]
    
    def size(file):
        filename = os.path.join(path,file)
        img = Image.open(filename)
        return img.size
    
    for dir in dirlist:
        
        path = root + '/' + dir
    
        onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    
        onlyfiles.sort(reverse = True ,key = size)
    
        time_captured = list(dict.fromkeys([w[0:8] for w in onlyfiles]))
    
        files2keep = []
    
        for fig in time_captured:
            index = [idx for idx, s in enumerate(onlyfiles) if fig in s][0]
            files2keep.append(onlyfiles[index])
    
        files2delete = [x for x in onlyfiles if x not in files2keep]
        
        [os.remove(os.path.join(path,file2delete)) for file2delete in files2delete]


    
#[delete_excess(carpeta) for carpeta in carpetas]

images = [os.path.join(dirpath,f) for (dirpath, dirnames, filenames) in os.walk('grietas') for f in filenames]

checkpoint_path = "/home/emanuelacm/tesis/cod_1/saved_models/checkpoint.pth"

def segmentation(checkpoint_path,images,pixel_h,pixel_w):
    
    df = pd.DataFrame(columns=['ID','Largo','Espesor','Path'])
    
    #  Load model 
    device = torch.device('cuda')
    model = build_unet()
    
    model = model.to(device)

    model.load_state_dict(torch.load(checkpoint_path,map_location = device))
    model.eval()

    for path in images:

        frame = cv2.imread(path)
        H, W, _ = frame.shape

        ori_frame = frame

        patch_size = 448

        H_r = int(H/patch_size + 1) * patch_size
        W_r = int(W/patch_size + 1) * patch_size

        frame = cv2.resize(frame, (W_r,H_r))

        frame_pred = frame[:,:,1]

        frame_pred = np.expand_dims(frame_pred, axis=-1)

        for r in range(0,frame.shape[0],patch_size):
            for c in range(0,frame.shape[1],patch_size):
                single_patch = frame[r:r+patch_size, c:c+patch_size,:]        
                single_patch = np.transpose(single_patch, ([2, 0, 1]))
                single_patch = single_patch/255.0
                single_patch = np.expand_dims(single_patch, axis=0)
                single_patch = single_patch.astype(np.float32)
                single_patch = torch.from_numpy(single_patch)
                single_patch = single_patch.to(device)

                with torch.no_grad():
                    pred_y = model(single_patch)
                    pred_y = torch.sigmoid(pred_y)
                    pred_y = pred_y[0].cpu().numpy()       
                    pred_y = np.squeeze(pred_y, axis=0)
                    pred_y = pred_y > 0.4
                    pred_y = np.array(pred_y, dtype=np.uint8) * 255
                    pred_y = np.expand_dims(pred_y, axis=-1)
                    frame_pred[r:r+patch_size, c:c+patch_size,:] = pred_y[:,:,:]

        frame_pred = cv2.resize(frame_pred, (W, H))
        
        #frame_pred = morphology.remove_small_objects(frame_pred, 50)
        #frame_pred = morphology.remove_small_holes(frame_pred, 50)    

        label_img = label(frame_pred) 
        
        contours = cv2.findContours(frame_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        
        frame_pred = cv2.cvtColor(frame_pred,cv2.COLOR_GRAY2BGR)
        
        for i in contours:
            x,y,w,h = cv2.boundingRect(i)
            cv2.rectangle(frame_pred, (x, y), (x + w, y + h), (255,0,0), 4)

        props = regionprops_table(label_img, properties=('axis_major_length',
                                                         'axis_minor_length'))
        
        df_props = pd.DataFrame(props)
        
        df_props.fillna(0)
        
    
        if(path.split('/')[1]=='grietas_transversales'):
            
            
            largo = int(np.nan_to_num(df_props['axis_major_length'].sum()))*pixel_w

            espesor = int(np.nan_to_num(df_props['axis_minor_length'].sum()))*pixel_h
            
            id = path.split('/')[2]
            df.loc[len(df.index)] = [int(id), largo, espesor,path]
            combine_frame = np.concatenate((ori_frame, frame_pred), axis=0)
            
        else:
            largo = int(np.nan_to_num(df_props['axis_major_length'].sum()))*pixel_h
                
            espesor = int(np.nan_to_num(df_props['axis_minor_length'].sum()))*pixel_w
                  
            id = path.split('/')[2]
            df.loc[len(df.index)] = [int(id), largo, espesor,path]
            combine_frame = np.concatenate((ori_frame, frame_pred), axis=1)
            
        combine_frame = combine_frame.astype(np.uint8)
        cv2.imwrite(path, combine_frame)
        
    return df


