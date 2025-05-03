def load_images_from_folder(folder):
    images = []
    labels = []
    class_names = os.listdir(folder)
    
    for class_name in class_names:
        class_folder = os.path.join(folder, class_name)
        if os.path.isdir(class_folder):
            for filename in os.listdir(class_folder):
                img_path = os.path.join(class_folder, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(img)
                    labels.append(class_name)
    
    return images, labels

def preprocess_images(images, target_size=(224, 224)):
    processed_images = []
    for img in images:
        img = cv2.resize(img, target_size)
        img = img / 255.0  # Normalize to [0, 1]
        processed_images.append(img)
    return np.array(processed_images)

def load_data(train_folder, test_folder):
    train_images, train_labels = load_images_from_folder(train_folder)
    test_images, test_labels = load_images_from_folder(test_folder)
    
    train_images = preprocess_images(train_images)
    test_images = preprocess_images(test_images)
    
    return (train_images, train_labels), (test_images, test_labels)