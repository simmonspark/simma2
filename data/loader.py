import numpy as np
def load_numpy_file(file_path):
    try:
        data = np.load(file_path)
        print(f"Loaded numpy file from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading numpy file: {e}")
        return None