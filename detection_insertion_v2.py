import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import itertools
from tqdm import tqdm
import time
import sqlite3

input_path = "E:\\Cameras2\\"
file_paths = []

for dir_path, subdir_list, file_list in os.walk(input_path):
    for fname in file_list:
        full_path = os.path.join(dir_path, fname)
        #print(full_path)
        file_paths.append(full_path)

num_files = (len(file_paths))


net = cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg')

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

classes = open('coco.names').read().strip().split("\n")

layer_names = net.getLayerNames()

output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

colors = np.random.uniform(0, 255, size=(200, 3))

connection = sqlite3.connect('C:\\Users\\Enea\\Desktop\\Senior Project\\sp_demo.db')

cursor = connection.cursor()

cursor.execute('''CREATE TABLE IF NOT EXISTS detections (id integer PRIMARY KEY, road_name TEXT, road_id TEXT,
               image_path TEXT,timestamp TEXT,car_count INTEGER,person_count INTEGER)''')


def perform_detection(net, img, output_layers, w, h, confidence_threshold):

    blob = cv2.dnn.blobFromImage(img, 1 / 255., (512, 512), swapRB=True, crop=False)

    net.setInput(blob)

    start = time.time()

    layer_outputs = net.forward(output_layers)

    end = time.time()


    print(" [INFO] Dectection: {:.6f} s".format(end - start))


    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Object is deemed to be detected
            if confidence > confidence_threshold:

                center_x, center_y, width, height = list(map(int, detection[0:4] * [w, h, w, h]))


                top_left_x = int(center_x - (width / 2))
                top_left_y = int(center_y - (height / 2))

                boxes.append([top_left_x, top_left_y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)


    return boxes, confidences, class_ids

def draw_boxes(boxes, confidences, class_ids, classes, img, colors, confidence_threshold, NMS_threshold):

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, NMS_threshold)

    FONT = cv2.FONT_HERSHEY_SIMPLEX

    total_vehicle_count = 0

    total_people_count = 0

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]

            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

            text = "{}: {:.4f}".format(classes[class_ids[i]], confidences[i])
            cv2.putText(img, text, (x, y - 5), FONT, 0.5, color, 2)

            list_of_vehicles = ["car","bus","truck"]

            if (classes[class_ids[i]] in list_of_vehicles):
                total_vehicle_count+=1
            elif (classes[class_ids[i]] =='person'):
                total_people_count+=1


    title = 'Cars:'+str(total_vehicle_count) +" People:"+str(total_people_count)

    plt.title(title)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    return total_vehicle_count,total_people_count

def detection_image_file(image_path,confidence_threshold, nms_threshold):

    img = cv2.imread(image_path)

    h,w,_= img.shape

    boxes, confidences, class_ids = perform_detection(net, img, output_layers, w, h, confidence_threshold)

    t_c , t_p = draw_boxes(boxes, confidences, class_ids, classes, img, colors, confidence_threshold, nms_threshold)

    return t_c,t_p



road_cameras = {}


file_path = "C:\\Users\\Enea\\Desktop\\Senior Project\\camera_ip_v2.txt"

with open(file_path) as file:

    for line1,line2 in itertools.zip_longest(*[file]*2):
        road_cameras[line1.rstrip('\n')] = line2.rstrip('\n')
     

cameras1=[]


for image_path in tqdm(file_paths[37700:], ncols = 60, position = 0):

    
     r_i=image_path.split('\\')[2]

     r_n = road_cameras[r_i]

     t_s = image_path.split('\\')[3].replace('.jpg', '')

     c_c,p_c = detection_image_file(image_path.replace('\\', '\\\\'),0.5,0.5)
     
     
     cameras1.append((None,r_n,r_i,image_path,t_s,c_c ,p_c))


for item in cameras1:

    cursor.execute('insert into detections values (?,?,?,?,?,?,?)',item)  
    connection.commit()
                    
connection.close()
