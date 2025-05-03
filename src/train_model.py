import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
from PIL import Image, UnidentifiedImageError

def safe_load_image(img_path):
    """
    Safely load an image, skipping corrupted files.
    """
    try:
        img = Image.open(img_path)
        img.verify()  # Verify that the image is not corrupted
        return True
    except (UnidentifiedImageError, OSError):
        return False

def create_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def train_model(train_dir, model_save_path, input_shape=(150, 150, 3), batch_size=32, epochs=10):
    # Filter out corrupted images
    for root, dirs, files in os.walk(train_dir):
        for file in files:
            img_path = os.path.join(root, file)
            if not safe_load_image(img_path):
                print(f"Skipping corrupted image: {img_path}")
                os.remove(img_path)  # Optionally remove the corrupted file

    # Prepare data generators
    train_datagen = ImageDataGenerator(rescale=1.0/255.0)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical'
    )

    num_classes = len(train_generator.class_indices)
    
    # Create model
    model = create_model(input_shape, num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Set up early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    # Train model
    model.fit(train_generator, epochs=epochs, callbacks=[early_stopping])

    # Save model
    model.save(model_save_path)

if __name__ == "__main__":
    train_model('C:/Users/Shishir/OneDrive/Documents/Yoga_AI/image-classification-ai/data/train', 'models/model.keras')