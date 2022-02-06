import requests
import numpy as np
import time as ti
from datetime import *
import os
import cv2
import itertools
import tqdm
from requests import ReadTimeout, ConnectTimeout, HTTPError, Timeout, ConnectionError
import matplotlib.pyplot as plt 


base_url = "http://infomobility.tirana.gov.al/images-cctv/"


road_cameras = {}


file_path = "C:\\Users\\Enea\\Desktop\\Senior Project\\camera_ip_v2.txt"

with open(file_path) as file:

    for line1,line2 in itertools.zip_longest(*[file]*2):
        road_cameras[line1.rstrip('\n')] = line2.rstrip('\n')


def save_image(img,road_id,time):

    file_name = str(datetime.strptime(time.replace(" GMT",""), "%a, %d %b %Y %H:%M:%S"))

    road_folder= road_id+'\\'

    cwd = 'E:\\CamerasNY\\'+road_folder

    save_path = cwd + file_name.replace(':','-').replace(' ','_') +'.jpg'


    if os.path.exists(cwd):
        cv2.imwrite(save_path,img)

    else:
        os.mkdir(cwd)
        cv2.imwrite(save_path,img)


def countdown(t): 
    
    while t: 
        mins, secs = divmod(t, 60) 
        timer = '{:02d}:{:02d}'.format(mins, secs) 
        print(timer, end="\n",flush=True) 
        ti.sleep(1) 
        t -= 1


#Main loop

count = 1
while True:

    print("RUN:",count)
    count+=1
    print('------------')
    
    x=0
    for road_id in road_cameras.keys():
        
        try:
            resp = requests.get(base_url+road_id+'.jpg', stream=True, timeout=10).raw

            image = np.asarray(bytearray(resp.read()), dtype="uint8")

            if image.size!=0 and resp.status==200 and image[-1]==217:
            
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)

                #save image in directory
                save_image(image,road_id,resp.headers['last-modified'])
                
                x+=1
                print("St. Name:",road_cameras[road_id]," Progress:",x,'/',len(road_cameras) )
                
                #comment to disable plots
                #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                #plt.axis('off')
                #plt.show()    
                
        except (ConnectTimeout, HTTPError, ReadTimeout, Timeout, ConnectionError):
            continue
            
    countdown(13)




