import tensorflow as tf
import numpy as np
import os
from pathlib import Path
import glob
import random

DATASET_PATH = r"D:\6S2\Thesis\7 dataset\hama1baru - RAW"

# =============================
# CONFIG
# =============================

MODEL_DIR = "models_h5"
OUTPUT_DIR = "models_tflite"

IMG_SIZE = 96
REPRESENTATIVE_SAMPLES = 200

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================
# REPRESENTATIVE DATASET
# =============================

def representative_dataset():

    raw_files = glob.glob(DATASET_PATH + r"\*\*.raw")

    random.shuffle(raw_files)

    raw_files = raw_files[:REPRESENTATIVE_SAMPLES]

    for f in raw_files:

        img = np.fromfile(f, dtype=np.uint8)

        img = img.reshape(IMG_SIZE, IMG_SIZE, 1)

        img = img.astype(np.float32)

        img = np.expand_dims(img, axis=0)

        yield [img]


# =============================
# CONVERT ONE MODEL
# =============================

def convert_model(h5_path):

    name = Path(h5_path).stem

    print("\n=============================")
    print("Converting:", name)
    print("=============================")

    model = tf.keras.models.load_model(h5_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    converter.representative_dataset = representative_dataset

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]

    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    # IMPORTANT for ESP32
    converter._experimental_disable_per_channel_quantization_for_dense_layers = True

    tflite_model = converter.convert()

    tflite_path = os.path.join(
        OUTPUT_DIR,
        f"{name}_INT8.tflite"
    )

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    print("TFLite saved:", tflite_path)

    convert_to_header(tflite_path, name)


# =============================
# TFLITE → HEADER
# =============================

def convert_to_header(tflite_path, name):

    with open(tflite_path, "rb") as f:
        data = f.read()

    header_path = os.path.join(
        OUTPUT_DIR,
        f"{name}_model.h"
    )

    with open(header_path, "w") as f:

        var_name = f"{name}_model"

        f.write(f"const unsigned char {var_name}[] = {{\n")

        for i, byte in enumerate(data):

            if i % 12 == 0:
                f.write("\n ")

            f.write(f"0x{byte:02x},")

        f.write("\n};\n")

        f.write(f"const unsigned int {var_name}_len = {len(data)};\n")

    print("Header saved:", header_path)


# =============================
# MAIN
# =============================

h5_files = [
    os.path.join(MODEL_DIR, f)
    for f in os.listdir(MODEL_DIR)
    if f.endswith(".h5")
]

print("Found models:", len(h5_files))

for model in h5_files:

    convert_model(model)

print("\nALL MODELS CONVERTED")