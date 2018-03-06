#!/usr/bin/env python
import os
import sys
import json

import cv2

from settings import (
    FRONTALFACE, PERSONS_PATH, RECOGNIZER_PATH, RECOGNIZER_THRESHOLD, GREEN
)


cam = cv2.VideoCapture(0)
recognizer = cv2.face.LBPHFaceRecognizer_create(threshold=RECOGNIZER_THRESHOLD)
classifier = cv2.CascadeClassifier(cv2.data.haarcascades + FRONTALFACE)


if __name__ == '__main__':
    if not os.path.exists(RECOGNIZER_PATH):
        print('\033[91mPlease train Recognizer model.\033[0m')
        sys.exit()

    recognizer.read(RECOGNIZER_PATH)
    persons = {}
    if os.path.exists(PERSONS_PATH):
        with open(PERSONS_PATH) as f:
            persons = json.load(f)

    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = classifier.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        for (x, y, w, h) in faces:
            label_predicted, confidence = recognizer.predict(
                gray[y:y + w, x:x + h]
            )
            name = persons.get(str(label_predicted))
            if name is None:
                name = 'Not detected'

            cv2.putText(
                frame, name, (x, y + h), cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN
            )

        cv2.imshow('Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
