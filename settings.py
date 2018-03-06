import os


SAMPLES_COUNT = 50
RECOGNIZER_THRESHOLD = 85

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

FRONTALFACE = 'haarcascade_frontalface_default.xml'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(BASE_DIR, 'samples')
PERSONS_PATH = os.path.join(BASE_DIR, 'persons.json')
RECOGNIZER_PATH = os.path.join(BASE_DIR, 'recognizer.yml')
