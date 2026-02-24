import os
import sys
import time
import numpy as np
import librosa
import pandas as pd
import logging

# =============================
# BASE PATH (quan trọng khi build exe)
# =============================
def get_base_path():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))

BASE_PATH = get_base_path()

DATA_FOLDER = os.path.join(BASE_PATH, "data2")
OUTPUT_FILE = os.path.join(BASE_PATH, "output2.xlsx")
LOG_FILE = os.path.join(BASE_PATH, "process.log")

MAX_VALID_HZ = 350
MIN_VALID_HZ = 75
TIME_MARKS = 20
SUB_WINDOWS = 5
PRE_SCAN_MARKS = 300


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
# DETECT PITCH
# =============================
def detect_pitch(segment, sr):
    try:
        f0 = librosa.yin(segment, fmin=50, fmax=500, sr=sr)
        f0 = f0[~np.isnan(f0)]
        if len(f0) == 0:
            return None
        return float(np.median(f0))
    except Exception as e:
        logger.error(f"Pitch detect error: {e}")
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
# PRE-SCAN 300 MARKS
# =============================
def prescan_300_marks(y, sr):

    total_len = len(y)
    segment_len = total_len // PRE_SCAN_MARKS

    prescan_data = []

    for i in range(PRE_SCAN_MARKS):
        start = i * segment_len
        end = (i + 1) * segment_len if i < PRE_SCAN_MARKS - 1 else total_len
        segment = y[start:end]

        hz = detect_pitch(segment, sr)
        midpoint_time = ((start + end) / 2) / sr

        prescan_data.append({
            "time": midpoint_time,
            "hz": hz
        })

    return prescan_data


# =============================
# FIND NEAREST VALID HZ
# =============================
def find_nearest_valid_hz(target_time, prescan_data):

    valid_points = [
        p for p in prescan_data
        if p["hz"] is not None and p["hz"] >= MIN_VALID_HZ
    ]

    if len(valid_points) == 0:
        return MIN_VALID_HZ

    nearest = min(valid_points, key=lambda p: abs(p["time"] - target_time))
    return nearest["hz"]


# =============================
# PROCESS FILE
# =============================
def process_audio_file(file_path):

    start_time_process = time.time()
    logger.info(f"Processing {os.path.basename(file_path)}")

    y, sr = librosa.load(file_path, sr=None)
    y = remove_silence(y)

    prescan_data = prescan_300_marks(y, sr)

    total_len = len(y)
    segment_len = total_len // TIME_MARKS

    hz_values = []
    segment_boundaries = []

    for i in range(TIME_MARKS):
        logger.info(f"  Segment {i+1}/{TIME_MARKS}")

        start = i * segment_len
        end = (i + 1) * segment_len if i < TIME_MARKS - 1 else total_len

        segment = y[start:end]
        hz = robust_pitch_detection(segment, sr)

        hz_values.append(hz)
        segment_boundaries.append((start, end))

    hz_values = smooth_hz_values(hz_values)

    results = []
    reset_time = 0
    prev_end_time = 0

    for i in range(TIME_MARKS):

        start, end = segment_boundaries[i]
        start_time = start / sr
        end_time = end / sr
        midpoint_time = (start_time + end_time) / 2

        if hz_values[i] < MIN_VALID_HZ:
            replacement_hz = find_nearest_valid_hz(midpoint_time, prescan_data)
            hz_values[i] = round(replacement_hz, 2)

        if i == 10:
            reset_time = prev_end_time

        if i < 10:
            relative_time = midpoint_time
            excel_mark = i + 1
        else:
            relative_time = midpoint_time - reset_time
            excel_mark = (i - 10) + 1

        results.append({
            "sounding": os.path.basename(file_path),
            "time_mark": excel_mark,
            "relative_time_sec": round(relative_time, 3),
            "hz": hz_values[i]
        })

        prev_end_time = end_time

    elapsed = time.time() - start_time_process
    logger.info(f"Finished in {elapsed:.2f}s\n")

    return results


# =============================
# MAIN
# =============================
def main():

    logger.info("===== START PROCESSING data2 =====")

    if not os.path.exists(DATA_FOLDER):
        logger.error("Folder 'data2' not found next to exe!")
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