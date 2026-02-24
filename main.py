import os
import sys
import numpy as np
import librosa
import pandas as pd
import logging

# =============================
# PATH HANDLING (quan trọng khi build exe)
# =============================
def get_base_path():
    if getattr(sys, 'frozen', False):
        # chạy từ exe
        return os.path.dirname(sys.executable)
    else:
        # chạy từ python
        return os.path.dirname(os.path.abspath(__file__))

BASE_PATH = get_base_path()
DATA_FOLDER = os.path.join(BASE_PATH, "data")
OUTPUT_FILE = os.path.join(BASE_PATH, "output.xlsx")
LOG_FILE = os.path.join(BASE_PATH, "process.log")

MAX_VALID_HZ = 350
TIME_MARKS = 10
SUB_WINDOWS = 5


# =============================
# LOGGING CONFIG
# =============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# =============================
# REMOVE SILENCE
# =============================
def remove_silence(y):
    y_trimmed, _ = librosa.effects.trim(y, top_db=30)
    return y_trimmed


# =============================
# DETECT HZ
# =============================
def detect_pitch(segment, sr):
    try:
        f0 = librosa.yin(segment, fmin=50, fmax=500, sr=sr)
        f0 = f0[~np.isnan(f0)]
        if len(f0) == 0:
            return None
        return float(np.median(f0))
    except Exception as e:
        logger.error(f"Pitch detection error: {e}")
        return None


# =============================
# ROBUST DETECT
# =============================
def robust_pitch_detection(segment, sr):

    hz = detect_pitch(segment, sr)

    if hz is not None and hz < MAX_VALID_HZ:
        return hz

    sub_len = len(segment) // SUB_WINDOWS
    valid_hz_values = []

    for i in range(SUB_WINDOWS):
        start = i * sub_len
        end = (i + 1) * sub_len if i < SUB_WINDOWS - 1 else len(segment)

        sub_segment = segment[start:end]
        sub_hz = detect_pitch(sub_segment, sr)

        if sub_hz is not None and sub_hz < MAX_VALID_HZ:
            valid_hz_values.append(sub_hz)

    if len(valid_hz_values) > 0:
        return float(np.median(valid_hz_values))

    return MAX_VALID_HZ


# =============================
# SMOOTHING
# =============================
def smooth_hz_values(hz_list):
    smoothed = []
    for i in range(len(hz_list)):
        if i == 0:
            value = np.mean(hz_list[0:2])
        elif i == len(hz_list) - 1:
            value = np.mean(hz_list[-2:])
        else:
            value = np.mean(hz_list[i-1:i+2])
        smoothed.append(round(float(value), 2))
    return smoothed


# =============================
# PROCESS FILE
# =============================
def process_audio_file(file_path):

    logger.info(f"Processing file: {os.path.basename(file_path)}")

    y, sr = librosa.load(file_path, sr=None)
    y = remove_silence(y)

    total_len = len(y)
    segment_len = total_len // TIME_MARKS

    hz_values = []
    segment_boundaries = []

    for i in range(TIME_MARKS):

        logger.info(f"  Processing segment {i+1}/{TIME_MARKS}")

        start_sample = i * segment_len
        end_sample = (i + 1) * segment_len if i < TIME_MARKS - 1 else total_len

        segment = y[start_sample:end_sample]
        hz = robust_pitch_detection(segment, sr)

        hz_values.append(hz)
        segment_boundaries.append((start_sample, end_sample))

    hz_values = smooth_hz_values(hz_values)

    results = []

    for i in range(TIME_MARKS):
        start_sample, end_sample = segment_boundaries[i]

        midpoint_time = ((start_sample + end_sample) / 2) / sr

        results.append({
            "sounding": os.path.basename(file_path),
            "time_mark": i + 1,
            "relative_time_sec": round(midpoint_time, 3),
            "hz": hz_values[i]
        })

    return results


# =============================
# MAIN
# =============================
def main():

    logger.info("===== START PROCESSING =====")

    if not os.path.exists(DATA_FOLDER):
        logger.error("Folder 'data' not found!")
        return

    all_results = []
    files = [f for f in os.listdir(DATA_FOLDER) if f.lower().endswith(".wav")]

    logger.info(f"Found {len(files)} wav files")

    for index, file_name in enumerate(files):
        logger.info(f"[{index+1}/{len(files)}] {file_name}")

        file_path = os.path.join(DATA_FOLDER, file_name)
        results = process_audio_file(file_path)
        all_results.extend(results)

    df = pd.DataFrame(all_results)
    df.to_excel(OUTPUT_FILE, index=False)

    logger.info("===== DONE =====")
    logger.info(f"Output saved to: {OUTPUT_FILE}")
    logger.info(f"Log saved to: {LOG_FILE}")


if __name__ == "__main__":
    main()