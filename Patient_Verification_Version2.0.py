import face_recognition
import cv2
import numpy as np
import os
import time
# single program for image loader 
# 'e' for esc , exit the program 

def hideWindow(name):
    global window 
    window = False
    try:
        cv2.destroyWindow(f'{name} detected')
    except:
        pass
video_capture = cv2.VideoCapture(0)
path = os.path.join('recognizer','persons')
face_encoding_data = {}



for file in os.listdir(path):
    print('loading', file)
    name = os.path.splitext(file)[0]
    person_image = f'{name}_image'
    img_path = os.path.join('recognizer','persons',file)
    person_face = face_recognition.load_image_file(img_path)
    person_face_encoding = face_recognition.face_encodings(person_face)[0]
    face_encoding_data[name]=person_face_encoding
print(face_encoding_data.keys())


known_face_encodings = list(face_encoding_data.values())
known_face_names = list(face_encoding_data.keys())


DELAY = 10
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

window = False
closing= 0
timestamp = 0

while True:
   
    ret, frame = video_capture.read()

   
    if process_this_frame:
       
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

       
        rgb_small_frame = small_frame[:, :, ::-1]
        
        
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

          
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        right *= 4 
        bottom *= 4
        left *= 4
        top *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        if not window:
            timestamp = time.time()
            window = True
            cv2.imshow(f'{name} detected',frame)
            closing = timestamp + DELAY
        if  closing <= round(time.time()):
            hideWindow(name)
        print(round(time.time()), closing, time.time() -  closing )

    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

video_capture.release()
cv2.destroyAllWindows()