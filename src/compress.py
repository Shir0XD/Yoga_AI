import gzip
import shutil
from tensorflow.keras.models import load_model


def compress_model(input_file, output_file):
    """
    Compress a model file using gzip.
    
    Args:
        input_file (str): Path to the input model file (e.g., 'model.h5').
        output_file (str): Path to the output compressed file (e.g., 'model.h5.gz').
    """
    with open(input_file, 'rb') as f_in:
        with gzip.open(output_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"Model compressed and saved as {output_file}")

# Load the model from the .h5 file
model = load_model('models/model.h5')

# Save the model in the .keras format
model.save('models/model.keras')

print("Model successfully converted from .h5 to .keras format.")

compress_model('models/model.keras', 'models/model.keras.gz')