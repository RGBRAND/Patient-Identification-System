from copyreg import pickle
import face_recognition
import numpy as np
import os
import pickle

path = os.path.join('recognizer','persons')
face_encoding_data = {}
# Load a sample picture and learn how to recognize it.
for file in os.listdir(path):
    print('loading', file)
    name = os.path.splitext(file)[0]
    person_image = f'{name}_image'
    img_path = os.path.join('recognizer','persons',file)
    person_face = face_recognition.load_image_file(img_path)
    person_face_encoding = face_recognition.face_encodings(person_face)[0]
    face_encoding_data[name]=person_face_encoding
print(face_encoding_data.keys())


with open('recognizer/faces.pk','wb') as f:
    pickle.dump(face_encoding_data, f)