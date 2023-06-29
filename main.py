
import cv2
import tensorflow as tf
import numpy as np
from keras.models import load_model


# Load the pre-trained model
model = load_model('my_model.h5', compile=False)

# Load the labels
labels = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

cap = cv2.VideoCapture(0)

while True:
    success, imgOrignal = cap.read()
    imgRGB = cv2.cvtColor(imgOrignal, cv2.COLOR_BGR2RGB)

    # Resize the image to the size required by the model
    imgRGB = cv2.resize(imgRGB, (180, 180))

    # Detect objects
    r = model.predict(np.expand_dims(imgRGB, axis=0))[0]


    # Draw bounding boxes and labels of the class predicted with the highest score
    cv2.putText(imgOrignal, labels[np.argmax(r)], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('imgOrignal', imgOrignal)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()