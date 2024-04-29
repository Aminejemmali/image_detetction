import numpy as np
import cv2
import cvlib as cv
from cvlib.face_detection import draw_bbox

def detect_features(frame, face_cascade, eye_cascade, mouth_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            center = (int(x + ex + 0.5*ew), int(y + ey + 0.5*eh))
            radius = int(0.3 * (ew + eh))
            cv2.circle(frame, center, radius, (0, 255, 0), 2)

        mouths = mouth_cascade.detectMultiScale(roi_gray)
        for (mx, my, mw, mh) in mouths:
            center = (int(x + mx + 0.5*mw), int(y + my + 0.5*mh))
            radius = int(0.3 * (mw + mh))
            cv2.circle(frame, center, radius, (0, 0, 255), 2)

        # Estimate age and gender
        faces, confidences = cv.detect_face(frame)
        for face in faces:
            (startX, startY) = face[0], face[1]
            (endX, endY) = face[2], face[3]
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            face_crop = np.copy(frame[startY:endY, startX:endX])
            (age, gender) = cv.detect_gender(face_crop)

            label = "{}, {}".format(age, gender)
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return frame

def main():
    face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
    eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
    mouth_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_mouth.xml')
    cap = cv2.VideoCapture(0)

    while(True):
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = detect_features(frame, face_cascade, eye_cascade, mouth_cascade)
        
        cv2.imshow('frame', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
