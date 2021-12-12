import numpy as np
import cv2
import pickle


face_cascade = cv2.CascadeClassifier(r'C:\Users\oscar\Documents\Universidad\Python\OpenCV Tutorial\cascades\data\haarcascade_frontalface_alt2.xml')
#eye_cascade = cv2.CascadeClassifier(r'C:\Users\oscar\Documents\Universidad\Python\OpenCV Tutorial\cascades\data\haarcascade_eye.xml')

#9.importamos el recognizer y el archivo .yml con el cual hicimos el entrenamiento
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")
#10. ahora vamos a cargar nuestro pickel que contiene nuestras etiquetas
labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)#1.seleccionamos la camara

#2. Vamos a crear un while, porque queremos que nuestra
#camara siga encendida de manera constante hasta que nosotros
#lo deseemos
while(True):
        
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        
        #5.Aqui vamos a hace la deteccion de rostro
        for (x, y, w, h) in faces:#donde empieza la x y hasta el alto y ancho
                
                roi_gray = gray[y:y+h, x:x+w] #(cod1->height, cord2->width)
                roi_color = frame[y:y+h, x:x+w]
                #11. Ahora vamos a predecir nuestras regiones de interes
                # usando nuestro recognizer
                id_, conf = recognizer.predict(roi_gray)
                #12. conf se refiere al nivel de confianza de la imagen
                # e importamos las etiquetas que vienen en el archivo pickle
                if conf>=45 and conf <= 85:
                        print(id_)
                        print(labels[id_])
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        name = labels[id_]
                        color = (255, 0, 255)
                        stroke = 2
                        #13. teneiendo ya lo anterior usamos putText para poner la informacion en el frame de la camara
                        cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
                
                img_item = "my-image.png"
                cv2.imwrite(img_item, roi_gray)

                #7.Dibujamos un rectangulo
                color = (250, 0, 0)
                stroke = 2 #grueso de la linea
                end_cord_x = x+w #ancho 
                end_cord_y = y+h #alto
                #8.creamos el rectangulo
                #tomamos el frame, luego las coordenadas iniciales, coordenadas finales, el color y el grosor
                cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
                #eyes = eye_cascade.detectMultiScale(roi_gray)
                #for(ex,ey,ew,eh) in eyes:
                        #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                
                
        #Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
                break

#When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
