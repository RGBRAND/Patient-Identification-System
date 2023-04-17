from platform import release
import cv2
import os

video = cv2.VideoCapture(0)
folder = 'recognizer'
face_model = f'{folder}/haarcascade_frontalface_default.xml'
facedetect = cv2.CascadeClassifier(face_model)

count = 0

nameID = str(input("Enter The Patient Name: ")).lower()

path = os.path.join(folder,'images',nameID)

isExist = os.path.exists(path)

if isExist:
    print("Name Already Taken")
    nameID = str(input("Enter Your Name Again: "))
else:
    os.makedirs(path)

while True:
    ret,frame = video.read()
    faces= facedetect.detectMultiScale(frame, 1.3, 5)
    for x,y,w,h in faces:
        count= count+1
        name = f'{folder}/images/'+nameID+'/'+str(count)+ '.jpg'
        print("Creating Images......"+name)
        cv2.imwrite(name, frame[y:y+h, x:x+w])
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
    cv2.imshow("WindowFrame", frame)
    k = cv2.waitKey(1)
    if count > 20:
        break

video.release()
cv2.destroyAllWindows()
