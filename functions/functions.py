import os
import pandas as pd
import folium as folium
import geopy.distance
import numpy as np
import cv2
import datetime

#===========================================================
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import geopandas as gpd
import json
from skimage.measure import label, regionprops, regionprops_table

#===========================================================

def video_metadata(video_path):
    cwd = os.getcwd()
    cmd_ffmpeg = 'ffmpeg -i ' + video_path + ' -map_metadata 0 > ' + cwd + '/output.txt 2>&1'
    os.system(cmd_ffmpeg)
    f = open('output.txt', "r")
    lines = f.readlines()

    duration = list(filter(lambda x: 'Duration' in x, lines))[0]
    duration = duration.replace(':',',')
    duration = duration.split(',')
    
    dur_hor = int(duration[1])
    dur_min = int(duration[2])
    dur_sec = int(round(float(duration[3])))
    
    fps = 60
    
    creation_time = list(filter(lambda x: 'creation_time' in x, lines))[0]
    creation_time = creation_time[creation_time.find('T')+len('T'):creation_time.find('.')]
    hour = creation_time[0:2]
    minutes = creation_time[3:5]
    seconds = creation_time[6:8]
    
    
    horarg = str(abs(int(hour) - dur_hor)).zfill(2)
    minutos = str(abs(int(minutes) - dur_min)).zfill(2)
    segundos = str(abs(int(seconds) - dur_sec)).zfill(2)
    
    tiempo_creacion = horarg + ':' + minutos + ':' + segundos
    
    print(tiempo_creacion)
    
    return fps, tiempo_creacion

def print_map(data_path,save_name):

    df_or = pd.read_csv(data_path, sep=',')

    df_or.head()

    df = df_or[['latitude', 'longitude']]

    route_map = folium.Map(
        location = df.iloc[int(len(df.iloc[:,0])/2),:],
        zoom_start=13,
        control_scale=True,
        tiles='cartodbdark_matter'
    )

    coordinates = [tuple(x) for x in df.to_numpy()]
    folium.PolyLine(coordinates, weight=6).add_to(route_map)

    route_map.save(outfile = save_name)
    
    return(route_map, df_or)

def frame_dimensions(video_path,pixel_h,pixel_w):
    
    vs = cv2.VideoCapture(video_path)
    _, frame = vs.read()
    H, W, _ = frame.shape
    vs.release()
    
    real_h = H * pixel_h 
    real_w = W * pixel_w
    
    return real_h,real_w



def dataframe_prep(path2gps_df):

    df_gps = pd.read_csv(path2gps_df)
    i = df_gps.columns.get_loc('date time')
    df2 = df_gps['date time'].str.split(" ", expand=True)
    df_gps = pd.concat([df_gps.iloc[:, :i], df2, df_gps.iloc[:, i+1:]], axis=1)
    df_gps.rename({0:'date', 1:'time'}, axis='columns', inplace = True)
    i = df_gps.columns.get_loc('time')
    df2 = df_gps['time'].str.split(".", expand=True)
    df_gps = pd.concat([df_gps.iloc[:, :i], df2, df_gps.iloc[:, i+1:]], axis=1)
    df_gps.rename({0:'time', 1:'resto'}, axis='columns', inplace = True)
    
    df_gps = df_gps[['time','latitude','longitude','speed(m/s)']]
    
    start_lat = df_gps.loc[0]['latitude']

    start_lon = df_gps.loc[0]['longitude']

    def distancer(row):
        coords_1 = (start_lat, start_lon)
        coords_2 = (row['latitude'], row['longitude'])
        return geopy.distance.geodesic(coords_1, coords_2).m

    df_gps['distance'] = df_gps.apply(distancer, axis=1)

    return df_gps

def detection_time(cnt_frames,fps_video,creation_time):

    secs = sum(int(x) * 60 ** i for i, x in enumerate(reversed(creation_time.split(':'))))
    
    segundos = int(cnt_frames / fps_video)
    
    frame_res = f"{(cnt_frames % fps_video):02}"

    time = str(datetime.timedelta(seconds=(secs+segundos)))
    
    return time,frame_res

#===============================================================================

def get_cutout(points, image):
    """Returns a rectangular cut-out from a set of points on an image"""
    max_x = int(max(points[:, 0]))
    min_x = int(min(points[:, 0]))
    max_y = int(max(points[:, 1]))
    min_y = int(min(points[:, 1]))
    
    return image[min_y:max_y, min_x:max_x]


