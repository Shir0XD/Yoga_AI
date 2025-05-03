from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

class ImageClassifier:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.class_indices = self._get_class_indices()

    def _get_class_indices(self):
        # Assuming the classes are organized in subfolders in the 'data/train' directory
        base_dir = 'data/train'
        class_names = os.listdir(base_dir)
        return {class_name: idx for idx, class_name in enumerate(class_names)}

    def predict(self, img_path):
        img = image.load_img(img_path, target_size=(150, 150))  # Adjust size as needed
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image
        predictions = self.model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class = list(self.class_indices.keys())[predicted_class_index]
        return predicted_class

def main(img_path):
    classifier = ImageClassifier('models/model.h5')
    predicted_class = classifier.predict(img_path)
    print(f'The predicted class is: {predicted_class}')

if __name__ == '__main__':
    # Example usage: replace 'path_to_image.jpg' with the actual image path
    main('path_to_image.jpg')