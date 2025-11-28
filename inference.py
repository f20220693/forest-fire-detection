import os
import json
import time
import psutil
import librosa
import pandas as pd
import numpy as np
import soundfile as sf
import tflite_runtime.interpreter as tflite
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ------- CONFIGURATION -------
SAMPLE_RATE = 16000
PATH_FSC22_AUDIO = "./fsc22_audios"  # Path to raw FSC22 audio folder
PATH_FSC22_META = "fsc22_metadata.csv" # Path to metadata CSV
MODEL_FILE = "fire_classifier_int8.tflite" # Using the int8 model
YAMNET_FILE = "yamnet.tflite"
LABEL_MAP_FILE = "label_map.json"
TEST_SAMPLE_SIZE = 100  # Limit to 100 files as requested

# ------- LOAD RESOURCES -------
# Load Label Map
with open(LABEL_MAP_FILE, "r") as f:
    label_map = json.load(f)["classes"]

# Load Models
print(f"Loading Models: {YAMNET_FILE} & {MODEL_FILE}...")
yamnet = tflite.Interpreter(YAMNET_FILE)
yamnet.allocate_tensors()

classifier = tflite.Interpreter(MODEL_FILE)
classifier.allocate_tensors()

# ------- HELPER FUNCTIONS -------
def normalize_label(label):
    """
    Normalizes class labels to treat 'no fire', 'non fire', 'noise', etc. as the same class.
    """
    label = str(label).lower().strip().replace(" ", "_")
    
    # Map all negative variations to 'non_fire'
    if label in ["no_fire", "non_fire", "noise", "not_fire", "clear"]:
        return "non_fire"
    
    return label

def load_audio(path):
    """Loads audio, converts to mono, and resamples to 16kHz."""
    try:
        wav, sr = sf.read(path)
    except Exception as e:
        # Fallback to librosa if soundfile fails
        try:
            wav, sr = librosa.load(path, sr=None)
        except:
            return None

    # Convert to mono if stereo
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    # Resample if necessary
    if sr != SAMPLE_RATE:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)

    return wav.astype(np.float32)

def get_embeddings(audio):
    """Runs YAMNet to get 1024-dim embeddings."""
    inp = yamnet.get_input_details()[0]
    idx = inp["index"]

    yamnet.resize_tensor_input(idx, [len(audio)], strict=False)
    yamnet.allocate_tensors()
    yamnet.set_tensor(idx, audio)
    yamnet.invoke()

    outputs = yamnet.get_output_details()
    emb = None
    for o in outputs:
        if o["shape"][-1] == 1024:
            emb = yamnet.get_tensor(o["index"])
            break

    if emb is None: return None
    
    # Global Average Pooling
    emb = np.mean(emb, axis=0).astype(np.float32).reshape(1, 1024)
    return emb

def read_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return int(f.read()) / 1000.0
    except:
        return None

# ------- PREDICTION LOGIC -------
def predict_file(path):
    audio = load_audio(path)
    if audio is None: return None

    # Performance snapshot
    start_time = time.time()
    
    # 1. Get Embeddings
    # This result is still a FLOAT32 array (1, 1024)
    emb = get_embeddings(audio)
    if emb is None: return None

    # 2. Run Classifier
    cls_in = classifier.get_input_details()[0]
    idx = cls_in["index"]
    
    # --- FIX: QUANTIZE INPUT FOR INT8 MODEL ---
    # The INT8 model expects data type INT8.
    if cls_in['dtype'] == np.int8:
        # Get quantization parameters
        scale, zero_point = cls_in['quantization']
        
        # Quantize the FLOAT32 embedding to INT8
        # Formula: quantized_value = np.round(float_value / scale + zero_point)
        # We also need to clip the values to fit within the INT8 range (-128 to 127)
        quantized_emb = np.int8(np.clip(
            (emb / scale) + zero_point, 
            -128, 127
        ))
        
        input_data = quantized_emb
    else:
        # If the model is not INT8 (e.g., FLOAT32 or FLOAT16), use the raw embedding
        input_data = emb
    # ----------------------------------------

    classifier.resize_tensor_input(idx, input_data.shape, strict=False)
    classifier.allocate_tensors()
    classifier.set_tensor(idx, input_data) # Use the correctly typed input_data
    classifier.invoke()

    # 3. Get Result
    out = classifier.get_tensor(classifier.get_output_details()[0]["index"])[0]
    
    # --- IMPORTANT: DEQUANTIZE OUTPUT IF THE OUTPUT IS ALSO QUANTIZED ---
    cls_out = classifier.get_output_details()[0]
    if cls_out['dtype'] == np.int8:
        scale, zero_point = cls_out['quantization']
        # Formula: float_value = scale * (quantized_value - zero_point)
        out = (out.astype(np.float32) - zero_point) * scale
    # ------------------------------------------------------------------

    pred_idx = np.argmax(out)
    raw_pred_label = label_map[pred_idx]
    confidence = out[pred_idx]

    inference_time = (time.time() - start_time) * 1000  # ms
    
    return {
        "raw_prediction": raw_pred_label,
        "confidence": confidence,
        "time_ms": inference_time,
        "cpu": psutil.cpu_percent(),
        "temp": read_cpu_temp()
    }

# ------- MAIN EXECUTION -------
if __name__ == "__main__":
    if not os.path.exists(PATH_FSC22_META):
        print(f"ERROR: Metadata file not found at {PATH_FSC22_META}")
        exit()

    print(f"\nReading metadata from {PATH_FSC22_META}...")
    df = pd.read_csv(PATH_FSC22_META)
    df.columns = df.columns.str.strip()

    # --- 1. FILTER & SAMPLE (Limit to 100) ---
    print(f"Total files in CSV: {len(df)}")
    
    # Shuffle and pick top 100 to get a random mix of Fire/Non-Fire
    df = df.sample(frac=1, random_state=42).head(TEST_SAMPLE_SIZE)
    print(f"Selected {len(df)} random files for testing.\n")

    y_true = []
    y_pred = []
    perf_times = []
    
    print(f"{'FILENAME':<45} | {'TRUE':<10} | {'PRED':<10} | {'TIME (ms)'}")
    print("-" * 85)

    test_start_time = time.time()

    for index, row in df.iterrows():
        fname = row.get('Dataset File Name')
        raw_class = str(row.get('Class Name', row.get('class', ''))).strip()
        full_path = os.path.join(PATH_FSC22_AUDIO, fname)

        if not os.path.exists(full_path):
            continue

        # --- 2. NORMALIZE TRUE LABEL ---
        # Logic: If CSV says 'Fire', it's fire. Everything else is 'non_fire'.
        if raw_class == 'Fire':
            true_label = 'fire'
        else:
            true_label = 'non_fire'

        # Run Prediction
        res = predict_file(full_path)
        if res is None: continue
            
        # --- 3. NORMALIZE PREDICTED LABEL ---
        predicted_label = normalize_label(res['raw_prediction'])
        
        # Store
        y_true.append(true_label)
        y_pred.append(predicted_label)
        perf_times.append(res['time_ms'])

        match_mark = "✔" if true_label == predicted_label else "✘"
        print(f"{fname[:43]:<45} | {true_label:<10} | {predicted_label:<10} | {res['time_ms']:.0f} {match_mark}")

    total_test_duration = time.time() - test_start_time

    # ------- FINAL REPORT -------
    print("\n\n" + "="*30)
    print("   TIME METRICS")
    print("="*30)
    print(f"Total Test Duration:  {total_test_duration:.2f} sec")
    print(f"Processed Files:      {len(perf_times)}")
    print(f"Avg Inference Time:   {np.mean(perf_times):.2f} ms")
    print(f"Min Inference Time:   {np.min(perf_times):.2f} ms")
    print(f"Max Inference Time:   {np.max(perf_times):.2f} ms")

    print("\n" + "="*30)
    print("   ACCURACY REPORT")
    print("="*30)
    
    acc = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_true, y_pred, digits=4))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("="*30)