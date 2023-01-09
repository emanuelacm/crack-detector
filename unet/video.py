import os
import numpy as np
import cv2
import torch

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

    # model = tf.keras.models.load_model("unet.h5")

    """ Reading frames """
    vs = cv2.VideoCapture(video_path)
    _, frame = vs.read()
    H, W, _ = frame.shape
    vs.release()

    H_v = int(H / 2)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('ideo_output.mp4',fourcc, 15.0,(W,H_v))

    #fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    #out = cv2.VideoWriter('output.avi', fourcc, 10, (W, H), True)

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
        frame = cv2.resize(frame, size)        
        frame = np.transpose(frame, (2, 0, 1))      ## (3, 512, 512)
        frame = frame/255.0
        frame = np.expand_dims(frame, axis=0)
        frame = frame.astype(np.float32)
        frame = torch.from_numpy(frame)
        frame = frame.to(device)
        with torch.no_grad():
            
            pred_y = model(frame)
            pred_y = torch.sigmoid(pred_y)
            pred_y = pred_y[0].cpu().numpy()        ## (1, 512, 512)
            pred_y = np.squeeze(pred_y, axis=0)     ## (512, 512)
            pred_y = pred_y > 0.4
            pred_y = np.array(pred_y, dtype=np.uint8) * 255
            print(pred_y.shape) 
            pred_y = np.expand_dims(pred_y, axis=-1)
            print(pred_y.shape)
            pred_y = cv2.resize(pred_y, (W, H))

        pred_y = cv2.cvtColor(pred_y,cv2.COLOR_GRAY2BGR)
        combine_frame = np.concatenate((ori_frame, pred_y), axis=1) 
        combine_frame = combine_frame.astype(np.uint8)
        H_c, W_c, _ = combine_frame.shape
        H_c = int(H_c * 0.5)
        W_c = int(W_c * 0.5)
        combine_frame = cv2.resize(combine_frame,(W_c, H_c))        
        
        cv2.imwrite(f"video/{idx}.png", combine_frame)
        idx += 1
        
        out.write(combine_frame)
        
        # Display the resulting frame
        cv2.imshow('Frame',combine_frame)
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break


cmd_ffmpeg = 'ffmpeg -i video_output.mp4 video_output.mp4'

os.system(cmd_ffmpeg)
