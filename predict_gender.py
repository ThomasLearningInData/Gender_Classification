import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Load the model
loaded_model = tf.keras.models.load_model('gender_detection.h5')

test_image_dir = 'Test_Image'
image_paths = [os.path.join(test_image_dir, f'test{i+1}.jpg') for i in range(5)]

fig, ax = plt.subplots(1, len(image_paths), figsize=(20, 5))

for i, image_path in enumerate(image_paths):
    img = Image.open(image_path)
    img_resized = img.resize((256, 256))  
    
    ax[i].imshow(img)
    ax[i].axis('off')  
    
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = loaded_model.predict(img_array)
    if prediction >= 0.8:
        label = 'Male'
    else:
        label = 'Female'

    # Show result
    ax[i].set_title(f'Prediction: {label}')

plt.show()