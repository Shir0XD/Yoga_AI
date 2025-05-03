<<<<<<< HEAD
# Yoga_AI
=======
# Image Classification AI

This project is an image classification AI model that classifies images based on folder names as classes. The model is trained using images organized in subfolders, where each subfolder name represents a class label.

## Project Structure

```
image-classification-ai
├── data
│   ├── train          # Training images organized in subfolders by class
│   └── test           # Test images organized in subfolders by class
├── models
│   └── model.h5      # Trained AI model in HDF5 format
├── src
│   ├── app.py        # Main application entry point for image uploads and predictions
│   ├── train_model.py # Code to train the AI model
│   ├── predict.py     # Functionality to make predictions on uploaded images
│   └── utils
│       └── data_loader.py # Utility functions for loading and preprocessing image data
├── requirements.txt   # List of dependencies required for the project
├── .gitignore         # Files and directories to be ignored by Git
└── README.md          # Documentation for the project
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd image-classification-ai
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Prepare your dataset:
   - Place your training images in the `data/train` directory, organized in subfolders by class.
   - Place your test images in the `data/test` directory, organized similarly.

4. Train the model:
   ```
   python src/train_model.py
   ```

5. Run the application:
   ```
   python src/app.py
   ```

## Usage

- Access the web application in your browser.
- Upload an image to classify it based on the trained model.
- The predicted class will be displayed after processing the image.

## License

This project is licensed under the MIT License.
>>>>>>> 6dae06b (Initial commit)
