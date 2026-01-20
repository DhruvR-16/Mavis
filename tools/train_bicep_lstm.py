import pandas as pd
import numpy as np
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle

# Robust Keras Import
try:
    from tensorflow import keras
except ImportError:
    try:
        import keras
    except ImportError:
        raise ImportError("Could not import keras from tensorflow or as standalone.")

# Constants

# Constants
SEQUENCE_LENGTH = 30
# Features based on CSV header:
FEATURES = [
    'Shoulder_Angle', 'Elbow_Angle', 'Hip_Angle', 'Knee_Angle', 'Ankle_Angle',
    'Shoulder_Ground_Angle', 'Elbow_Ground_Angle', 'Hip_Ground_Angle', 'Knee_Ground_Angle', 'Ankle_Ground_Angle'
]

def load_data(csv_path):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Filter columns
    X_data = df[FEATURES].values
    y_data = df['Label'].values
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_data)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_data)
    
    return X_scaled, y_encoded, le, scaler

def create_sequences(X, y, time_steps=SEQUENCE_LENGTH, step=1):
    X_seq, y_seq = [], []
    for i in range(0, len(X) - time_steps, step):
        X_seq.append(X[i : i + time_steps])
        y_seq.append(y[i + time_steps - 1]) 
    return np.array(X_seq), np.array(y_seq)

def train_model():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'bicep_angles.csv')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    # 1. Prepare Data
    X_scaled, y_encoded, le, scaler = load_data(DATA_PATH)
    
    print("Creating sequences...")
    X, y = create_sequences(X_scaled, y_encoded)
    print(f"Input Shape: {X.shape}")
    
    num_classes = len(np.unique(y_encoded))
    y_cat = keras.utils.to_categorical(y, num_classes=num_classes)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

    # 2. Build Model
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(64, return_sequences=True, activation='relu', input_shape=(SEQUENCE_LENGTH, len(FEATURES))))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.LSTM(128, return_sequences=False, activation='relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 3. Train
    print("Training model...")
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    # 4. Save Artifacts
    model.save(os.path.join(MODELS_DIR, 'bicep_lstm.h5'))
    
    with open(os.path.join(MODELS_DIR, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(le, f)
        
    with open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
        
    print("Model, Label Encoder, and Scaler saved to models/")
    print("Classes:", le.classes_)

if __name__ == "__main__":
    train_model()
