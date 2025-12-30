import os
import glob
import numpy as np
import scipy.sparse as sp
import scipy.io as sio

# CONFIGURATION
BASE_DIR = r"C:\Users\wyattluke.lowery\Documents\GitHub\sparse-sgwt\sgwt\library\data"
TARGET_DIRS = ["DELAY", "IMPEDANCE", "LENGTH"]
SIGNAL_DIR = "SIGNALS"

def convert_to_mat(directory):
    """Converts Sparse Matrix .npz files to .mat"""
    npz_files = glob.glob(os.path.join(directory, "*.npz"))
    
    for filepath in npz_files:
        try:
            # Load and convert to CSC (Sparse)
            matrix = sp.load_npz(filepath).tocsc()
            
            # Save as .mat
            mat_path = os.path.splitext(filepath)[0] + ".mat"
            sio.savemat(mat_path, {"A": matrix})
            
            print(f"Saved Matrix: {os.path.basename(mat_path)}")
        except Exception as e:
            print(f"Error {filepath}: {e}")

def convert_signals_to_mat(directory):
    """Converts Dense Signal .npz files (lat/lon) to .mat"""
    npz_files = glob.glob(os.path.join(directory, "*.npz"))
    
    for filepath in npz_files:
        try:
            # Load numpy archive
            data = np.load(filepath)
            
            # Extract specific keys based on your requirements
            if 'longitude' in data and 'latitude' in data:
                out_dict = {
                    "longitude": data['longitude'],
                    "latitude": data['latitude']
                }
                
                mat_path = os.path.splitext(filepath)[0] + ".mat"
                sio.savemat(mat_path, out_dict)
                
                print(f"Saved Signal: {os.path.basename(mat_path)}")
            else:
                print(f"Skipping {os.path.basename(filepath)}: Missing lat/lon keys")
                
        except Exception as e:
            print(f"Error {filepath}: {e}")

if __name__ == "__main__":
    # 1. Convert Sparse Matrices
    for folder in TARGET_DIRS:
        convert_to_mat(os.path.join(BASE_DIR, folder))

    # 2. Convert Signals
    convert_signals_to_mat(os.path.join(BASE_DIR, SIGNAL_DIR))