import os
import numpy as np
import cv2
import torch

from model import build_unet


if __name__ == "__main__":

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

    """ Video Path """
    video_path = "videos/20220730_172224.mp4"
    
    """ Reading frames """
    vs = cv2.VideoCapture(video_path)
    _, frame = vs.read()
    H, W, _ = frame.shape
    vs.release()

    H_v = int(H / 2)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('video_output4K.mp4',fourcc, 15.0,(W,H_v))

    cap = cv2.VideoCapture(video_path)
    
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            cap.release()
            out.release()
            break
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
                    pred_y = pred_y[0].cpu().numpy()        ## (1, 512, 512)
                    pred_y = np.squeeze(pred_y, axis=0)     ## (512, 512)
                    pred_y = pred_y > 0.4
                    pred_y = np.array(pred_y, dtype=np.uint8) * 255
                    pred_y = np.expand_dims(pred_y, axis=-1)
                    frame_pred[r:r+patch_size, c:c+patch_size,:] = pred_y[:,:,:]
                
        frame_pred = cv2.resize(frame_pred, (W, H))    
        frame_pred = cv2.cvtColor(frame_pred,cv2.COLOR_GRAY2BGR)
        combine_frame = np.concatenate((ori_frame, frame_pred), axis=1) 
        combine_frame = combine_frame.astype(np.uint8)
        H_c, W_c, _ = combine_frame.shape
        H_c = int(H_c * 0.5)
        W_c = int(W_c * 0.5)
        combine_frame = cv2.resize(combine_frame,(W_c, H_c))    
        out.write(combine_frame)
        
        # Display the resulting frame
        cv2.imshow('Frame',combine_frame)
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break


cmd_ffmpeg = 'ffmpeg -y -i video_output4K.mp4 video_output4K.mp4'

os.system(cmd_ffmpeg)
