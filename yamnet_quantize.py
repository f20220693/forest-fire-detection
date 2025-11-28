import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa
import os
import glob

AUDIO_DIR = "./fsc22_audios" 
OUTPUT_MODEL_NAME = "yamnet_int8.tflite"
YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
CALIBRATION_STEPS = 50  # Number of audio chunks to use for calibration

def load_yamnet_model():
    """Loads the YAMNet model from TensorFlow Hub."""
    print("Download/Loading YAMNet from TF Hub...")

    model = tf.keras.Sequential([
        hub.KerasLayer(YAMNET_URL, trainable=False)
    ])
    model.build(input_shape=(1, 15600)) 
    return model

def representative_dataset_gen():
    """
    Generator function that yields audio data to calibrate the model.
    It simulates what the model will see during real inference.
    """
    audio_files = glob.glob(os.path.join(AUDIO_DIR, "*.wav"))
    
    if not audio_files:
        raise ValueError(f"No .wav files found in {AUDIO_DIR}. Please check the path.")

    print(f"Found {len(audio_files)} files for calibration. Using {CALIBRATION_STEPS} samples.")
    
    count = 0
    for audio_path in audio_files:
        # Load audio at 16kHz (YAMNet requirement)
        wav, _ = librosa.load(audio_path, sr=16000)
        
        # Normalize to -1.0 to 1.0 (Standard audio range)
        if len(wav) > 0:
            max_val = np.max(np.abs(wav))
            if max_val > 0:
                wav = wav / max_val
        
        # YAMNet expects chunks of 15600 samples (0.975 seconds)
        # We will yield a few chunks from each file
        step = 15600
        for i in range(0, len(wav) - step, step):
            if count >= CALIBRATION_STEPS:
                return

            chunk = wav[i : i + step]
            # Reshape to (1, 15600) and ensure float32
            yield [np.array(chunk, dtype=np.float32).reshape(1, 15600)]
            count += 1

def quantize_and_save(model):
    print("Setting up TFLite Converter with Int8 quantization...")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # 1. Enable standard optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # 2. Provide the representative dataset for calibration
    converter.representative_dataset = representative_dataset_gen
    
    # 3. Restrict operations to Int8 (Validation ensures no Float ops remain)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    # 4. Enforce Input/Output to be Integer (Full Integer Quantization)
    # This makes the input tensor require int8/uint8 instead of float32
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8  
    
    print("Converting model... (This may take a minute)")
    tflite_model = converter.convert()
    
    # Save the file
    with open(OUTPUT_MODEL_NAME, "wb") as f:
        f.write(tflite_model)
    
    print(f"Success! Model saved to: {OUTPUT_MODEL_NAME}")
    print(f"Size: {len(tflite_model) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    try:
        model = load_yamnet_model()
        quantize_and_save(model)
    except Exception as e:
        print(f"Error: {e}")
