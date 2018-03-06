#!/usr/bin/env python
import os
import json

import cv2

from settings import (
    FRONTALFACE, GREEN, SAMPLES_COUNT, SAMPLES_DIR, PERSONS_PATH
)


cam = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier(cv2.data.haarcascades + FRONTALFACE)


def get_person_label():
    while True:
        label = raw_input('Enter your id (should be a number): ')
        if label.isdigit():
            break
        print('\033[93mID should be a number\033[0m')

    name = raw_input('Enter your full name: ')
    persons = {}
    if os.path.exists(PERSONS_PATH):
        with open(PERSONS_PATH) as f:
            persons = json.load(f)

    previous_name = persons.get(label)
    if previous_name is not None and previous_name != name:
        answers = {
            'positive': ('', 'Y', 'y', 'Yes', 'yes'),
            'negative': ('N', 'n', 'No', 'no')
        }
        while True:
            answer = raw_input(
                'Person with id = {} already known as `{}`, '
                'rewrite with new name? [Y/n]: '.format(label, previous_name)
            )
            if answer in answers['negative']:
                name = previous_name

            if answer in answers['positive'] or answer in answers['negative']:
                break

    persons[label] = name
    with open(PERSONS_PATH, 'w') as f:
        json.dump(persons, f)

    return int(label)


def capture_samples(label):
    if not os.path.exists(SAMPLES_DIR):
        os.makedirs(SAMPLES_DIR)

    label_path = os.path.join(SAMPLES_DIR, str(label))
    if not os.path.exists(label_path):
        os.makedirs(label_path)

    count = 0
    while count < SAMPLES_COUNT:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = classifier.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) == 1:
            x, y, w, h = faces[0]
            sample_path = os.path.join(
                label_path, '{}.{}.jpg'.format(label, count)
            )
            cv2.imwrite(sample_path, gray[y:y + w, x:x + h])
            cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN, 2)
            cv2.imshow('Capture', frame)
            cv2.waitKey(1)
            count += 1

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    capture_samples(get_person_label())
