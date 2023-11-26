from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import numpy as np
import os

from keras.models import load_model

EMOTIONS = np.array(['Angry', 'Disgusted', 'Evil', 'Fearful', 'Happy', 'Sad', 'Surprised'])
MODEL = None

TEST_IMG = 'Doodles/happy/1.png'
IMG_WIDTH  = 250
IMG_HEIGHT = 250

def predict_doodle(filename):

    image = load_img(filename, target_size=(IMG_WIDTH, IMG_HEIGHT))
    img = np.array(image)
    img = img / 255
    img = img.reshape(1, IMG_WIDTH, IMG_HEIGHT, 3)
    scores = MODEL.predict(img)

    print(f"Scores for {filename}:")
    for i, idx in enumerate(np.argsort(-scores[0]), start=1):
        print(f"({i}) {EMOTIONS[idx]}: \t {scores[0][idx]}")

if __name__ == '__main__':
    print("DoodlEmotion: Detection of emotions from simple doodles")
    
    print("Loading model... ", end='', flush=True)
    try:
        MODEL = load_model('doodlemotion_model.keras')
    except Exception:
        print("ERROR")
        raise Exception
    print("OK")
    
    print("Exit the program with Ctrl + C")

    while True:
        try:
            filename = input("\nInput an image: ")
            
            if os.path.exists(filename):
                predict_doodle(filename)
            else:
                print(f"File '{filename}' does not exist")

        except KeyboardInterrupt:
            print('\nInterrupted by user, exiting DoodlEmotion...')
            exit(1)

