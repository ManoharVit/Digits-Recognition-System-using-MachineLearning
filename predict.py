import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pyscreenshot as ScrCap
import time
import warnings
warnings.filterwarnings("ignore")

for i in range(0,3):
    time.sleep(8)
    im = ScrCap.grab(bbox=(60,440,740,1120)) #680 X 680
    image = im.resize((28, 28))
    print("Saved......", i)
    image.save('digit' + str(i) + '.png')
    print("Redraw........")


train_new_model = True
model = tf.keras.models.load_model('digits_reco.model')
image_number = 0
while os.path.isfile('C:/Users/chinn/Desktop/Digits Recognisation/digit{}.png'.format(image_number)):
    try:
        img = cv2.imread('digit{}.png'.format(image_number))[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print("The number is probably a {}".format(np.argmax(prediction)))
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
       # plt.show(block=False)
        # plt.pause(2)
        image_number += 1
    except:
        print("Error reading image "+str(image_number)+"..")
        image_number += 1

