import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Define directories
ecg_data_dir = 'ECG/train'  # Update with actual ECG dataset path
spect_data_dir = 'SPECT/train'  # Update with actual SPECT dataset path

# Image parameters
img_height, img_width = 224, 224
batch_size = 32

# Data Augmentation and Loading
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

def create_generator(directory, subset, class_mode):
    return datagen.flow_from_directory(
        directory,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode=class_mode,
        subset=subset
    )

# ECG Data Generators
train_ecg_generator = create_generator(ecg_data_dir, 'training', 'categorical')
validation_ecg_generator = create_generator(ecg_data_dir, 'validation', 'categorical')

# SPECT MPI Data Generators
train_spect_generator = create_generator(spect_data_dir, 'training', 'categorical')
validation_spect_generator = create_generator(spect_data_dir, 'validation', 'categorical')

# Print class mappings
print("ECG Class Mapping:", train_ecg_generator.class_indices)
print("SPECT Class Mapping:", train_spect_generator.class_indices)

# Define ECG Model
input_ecg = keras.Input(shape=(img_height, img_width, 3))
x = layers.Conv2D(32, (3,3), activation='relu')(input_ecg)
x = layers.MaxPooling2D(2,2)(x)
x = layers.Conv2D(64, (3,3), activation='relu')(x)
x = layers.MaxPooling2D(2,2)(x)
x = layers.Conv2D(128, (3,3), activation='relu')(x)
x = layers.MaxPooling2D(2,2)(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
ec_output = layers.Dense(4, activation='softmax')(x)  # 4 classes for ECG

model_ecg = keras.Model(inputs=input_ecg, outputs=ec_output)
model_ecg.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_ecg.fit(train_ecg_generator, validation_data=validation_ecg_generator, epochs=20)
model_ecg.save('ecg_model.h5')

# Define SPECT Model
input_spect = keras.Input(shape=(img_height, img_width, 3))
y = layers.Conv2D(32, (3,3), activation='relu')(input_spect)
y = layers.MaxPooling2D(2,2)(y)
y = layers.Conv2D(64, (3,3), activation='relu')(y)
y = layers.MaxPooling2D(2,2)(y)
y = layers.Conv2D(128, (3,3), activation='relu')(y)
y = layers.MaxPooling2D(2,2)(y)
y = layers.Flatten()(y)
y = layers.Dense(128, activation='relu')(y)
y = layers.Dropout(0.5)(y)
spect_output = layers.Dense(2, activation='softmax')(y)  # 2 classes for SPECT MPI

model_spect = keras.Model(inputs=input_spect, outputs=spect_output)
model_spect.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_spect.fit(train_spect_generator, validation_data=validation_spect_generator, epochs=20)
model_spect.save('spect_model.h5')

# Function to preprocess input image
def preprocess_image(image_path):
    from tensorflow.keras.preprocessing import image
    img = image.load_img(image_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Function to Predict using Both Models
def predict_final_result(ecg_image_path, spect_image_path):
    ecg_image = preprocess_image(ecg_image_path)
    spect_image = preprocess_image(spect_image_path)
    
    ecg_pred = model_ecg.predict(ecg_image)
    spect_pred = model_spect.predict(spect_image)
    
    ecg_class = np.argmax(ecg_pred)
    spect_class = np.argmax(spect_pred)
    
    if spect_class == 1:  # Abnormal SPECT MPI
        return 'Abnormal Heart Condition'
    elif ecg_class == 1:  # Myocardial Infarction ECG
        return 'Possible Myocardial Infarction'
    elif ecg_class == 2:  # History of MI ECG
        return 'History of Myocardial Infarction'
    elif ecg_class == 3:  # Abnormal Heartbeat
        return 'Abnormal Heartbeat Detected'
    else:
        return 'Normal Heart Condition'

# Example Usage
# print(predict_final_result('path_to_ecg_image', 'path_to_spect_image'))
