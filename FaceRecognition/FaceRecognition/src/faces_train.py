import cv2
import os
import numpy as np
from PIL import Image
import pickle

#1.importamos imagenes
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

#5. importamos nuestro clasificador
face_cascade = cv2.CascadeClassifier(r'C:\Users\oscar\Documents\Universidad\Python\OpenCV Tutorial\cascades\data\haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

#7
current_id = 0
label_ids = {}
y_labels = []
x_train = []

#2. Ahora vamos a ver las imagenes que hay alli
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            #3.tomamos el nombre del directorio
            label = os.path.basename(root).replace(" ", "-").lower()

            #8 Creamos la condicion            
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            #print(label_ids)
            #y_labels.append(label)
            #x_train.append(path)

#4. con las siguientes lineas convertimos las imagenes en un arreglo de numeros
#y además les damos el mismo tamaño
            pil_image = Image.open(path).convert("L") #primero los convertimos a una escala de grises
            size = (550, 550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(pil_image, "uint8")
            #print(image_array)

            #6. Faces sera el detector
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

            for(x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml")
