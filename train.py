#!/usr/bin/env python
import os

import cv2
import numpy as np

from settings import SAMPLES_DIR, FRONTALFACE, RECOGNIZER_PATH


recognizer = cv2.face.LBPHFaceRecognizer_create()
classifier = cv2.CascadeClassifier(cv2.data.haarcascades + FRONTALFACE)


def get_images_and_labels():
    images, labels = [], []
    for label in os.listdir(SAMPLES_DIR):
        label_dir_path = os.path.join(SAMPLES_DIR, label)
        image_paths = [
            os.path.join(label_dir_path, f) for f in os.listdir(label_dir_path)
            if os.path.isfile(os.path.join(label_dir_path, f))
        ]

        for path in image_paths:
            images.append(cv2.imread(path, 0))
            labels.append(int(os.path.split(path)[1].split('.')[0]))

    return images, labels


if __name__ == '__main__':
    images, labels = get_images_and_labels()

    if os.path.exists(RECOGNIZER_PATH):
        recognizer.read(RECOGNIZER_PATH)

    recognizer.update(images, np.array(labels))
    recognizer.write(RECOGNIZER_PATH)
