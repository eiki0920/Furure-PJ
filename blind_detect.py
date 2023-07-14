import os,sys
import cv2
import dlib
from imutils import face_utils
from scipy.spatial import distance
import datetime
import csv

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('../haarcascade_frontalface_alt2.xml')
face_parts_detector = dlib.shape_predictor('../shape_predictor_68_face_landmarks.dat')
blind_count = 0
prev_blink = False

# 目のアスペクト比を求める
def calc_ear(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    beside = distance.euclidean(eye[0], eye[3])
    vertical = (A + B) / 2
    eye_ear = vertical / beside
    return round(eye_ear, 3)


while True:
    # カメラから１フレーム読み込む
    ret, img = cap.read()

    #画像をグレースケール化
    img_gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # facesには顔の位置情報が格納されている
    faces = face_cascade.detectMultiScale(
            img_gry, scaleFactor=1.11, minNeighbors=3, minSize=(100, 100))

    if len(faces) == 1:
        x, y, w, h = faces[0, :]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        face = dlib.rectangle(x, y, x + w, y + h)
        face_parts = face_parts_detector(img_gry, face)
        face_parts = face_utils.shape_to_np(face_parts)

        # 右目のアスペクト比を求める
        right_eye = face_parts[42:48]
        right_eye_ear = calc_ear(right_eye)

        # 左目のアスペクト比を求める
        left_eye = face_parts[36:42]
        left_eye_ear = calc_ear(left_eye)

        if (right_eye_ear + left_eye_ear) < 0.4:
            curEyeOpen = 0
        else:
            curEyeOpen = 1
        
        if curEyeOpen == 0 and lastEyeOpen == 1:
            blind_count += 1

            with open('data.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)

                # csvファイルが空である場合はヘッダー行を書き込む
                if csvfile.tell() == 0:
                    writer.writerow([ 'Timestamp', 'Blink' ,'Blink_Count'])

                writer.writerow([str(datetime.datetime.now()), 'blink!!', blind_count])

        
        lastEyeOpen = curEyeOpen
            
            
    cv2.putText(img, str(blind_count), (10, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv2.LINE_AA)         
    cv2.imshow('frame', img)

    if cv2.waitKey(1) == 27:
        break  # esc to quit

cap.release()
cv2.destroyAllWindows()