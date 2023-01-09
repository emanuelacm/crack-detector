import argparse
from typing import List, Optional, Union
import os
import yolov5
import numpy as np
import torch
import torchvision.ops.boxes as bops
from functions.norfair_functions import *
import pandas as pd
import math

from functions.functions import *
from functions.maps import *
from functions.images import *

import norfair
from norfair import Detection, Tracker, Video, Paths

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
import json
import folium

DISTANCE_THRESHOLD_BBOX: float = 2.33
MAX_DISTANCE: int = 100000

current_dir = os.getcwd()

for file in os.listdir(current_dir + "/weights"):
    if file.endswith(".pt"):
        path = os.path.join(current_dir + "/weights", file)
    if file.endswith(".pth"):
        checkpoint_path = os.path.join(current_dir + "/weights", file)
        
for file in os.listdir(current_dir + "/data"):
    if file.endswith(".mp4"):
        video_path = os.path.join(current_dir + "/data", file)
    if file.endswith(".txt"):
        gps_path = os.path.join(current_dir + "/data", file)


model = YOLO(path)

video = Video(input_path = video_path)

distance_function = iou
distance_threshold = DISTANCE_THRESHOLD_BBOX

tracker = Tracker(
    distance_function=distance_function,
    distance_threshold=distance_threshold,
    initialization_delay=0,
)


pixel_h = 0.0006348867037195382

pixel_w = 0.0006241547903880162

df_ids = pd.DataFrame(columns=['ID','Espesor','Largo','Path'])

df_grietas = pd.DataFrame(columns=['Tipo'
                                   ,'ID'
                                   ,'T_det'
                                   ])

fps, creation_time = video_metadata(video_path)

df_gps = dataframe_prep(gps_path)

frame_cnt=0

root_dir = 'grietas'

carpetas = ['grietas_longitudinales','grietas_transversales','grietas_cocodrilo']

[os.makedirs(os.path.join(root_dir,carpeta), exist_ok=True) for carpeta in carpetas]

for frame in video:
    
    frame_cnt+=1
    
    yolo_detections = model(
        frame,
        conf_threshold=0.6,
        classes=[0,1,2]
    )
    
    tiempo_detect,cuadros_restantes = detection_time(frame_cnt,fps,creation_time)
    
    detections = yolo_detections_to_norfair_detections(yolo_detections)
    tracked_objects = tracker.update(detections=detections)
        
    for tracked in tracked_objects:
        df_grietas.loc[frame_cnt] = [int(tracked.label)
                              ,tracked.id
                              ,tiempo_detect
                              ]
        
        crop = get_cutout(tracked.estimate,frame)
        
        path_crop = root_dir + '/' + carpetas[tracked.label] + '/' + str(tracked.id)
        
        os.makedirs(path_crop, exist_ok=True) 
        
        image = path_crop + '/' + tiempo_detect.replace(':','_') + '_' + str(cuadros_restantes) + '.png'
        
        cv2.imwrite(image, crop)
            
    norfair.draw_boxes(frame, detections)
    norfair.draw_tracked_boxes(frame, tracked_objects,color_by_label=True,draw_labels=False)
    video.show(frame)
    video.write(frame)
    

df_grietas = pd.merge(df_grietas,df_gps,how='outer',left_on='T_det',right_on='time')

df_grietas.drop(['T_det','speed(m/s)','distance','time'], inplace=True, axis=1)

df_grietas = df_grietas.drop_duplicates()

df_grietas.to_csv('grietas.csv', encoding='utf-8', index=False)

[delete_excess(root_dir + '/' + carpeta) for carpeta in carpetas]

images = [os.path.join(dirpath,f) for (dirpath, dirnames, filenames) in os.walk('grietas') for f in filenames]

df_carac = segmentation(checkpoint_path,images,pixel_h,pixel_w)

df_carac.sort_values(by=['ID'])

df_carac.to_csv('caracteristicas.csv', encoding='utf-8', index=False)

df_caracs = pd.merge(df_grietas,df_carac,on='ID')

df_caracs.to_csv('caracteristicas_merge.csv', encoding='utf-8', index=False)

df_caracs.drop(['latitude','longitude'], inplace=True, axis=1)

df_long = df_caracs.query('Tipo==0.0')

df_long = df_long.groupby('ID',as_index=False).agg({'Tipo':'first','Espesor':'mean','Largo':'sum','Path':'first'})

df_long.to_csv('df_long.csv', encoding='utf-8', index=False)

df_resto = df_caracs.query('Tipo!=0.0')

df_resto = df_resto.groupby('ID',as_index=False).agg({'Tipo':'first','Espesor':'mean','Largo':'mean','Path':'first'})

df_resto.to_csv('df_resto.csv', encoding='utf-8', index=False)

df_caracs = pd.concat([df_long,df_resto])

df_caracs[['Espesor','Largo','Path']] =  df_caracs[['Espesor','Largo','Path']].fillna('-')

df_caracs.to_csv('carateristicas.csv', encoding='utf-8', index=False)

df_caracs.sort_values(by=['ID'])

save_name = 'ubicacion_grietas.html'

df_caracs = df_caracs.drop_duplicates()

generar_mapa(df_grietas,df_caracs,save_name)

df_gps.to_csv('gps.csv', encoding='utf-8', index=False)

