import serial
import time
import numpy as np
import os
import random
import csv
from datetime import datetime
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# =============================
# CONFIG
# =============================

PORT = "COM3"
BAUD = 115200

IMG_SIZE = 96
IMG_BYTES = IMG_SIZE * IMG_SIZE

# NUM_TEST = 1000
# PERIOD = 2

DATASET_PATH = r"D:\6S2\Thesis\7 dataset\hama3_split_raw\test"
os.makedirs("results", exist_ok=True)

# MODEL = "B5_4_4_MobileNetV1_model"
# MODEL = input("Masukkan nama model: ").strip()

# ITERASI INPUT

default_iter = 100

inp = input(f"Berapa Iterasi [{default_iter}]: ").strip()

try:
    NUM_TEST = int(inp) if inp else default_iter
except ValueError:
    print("Input tidak valid, menggunakan default")
    NUM_TEST = default_iter

# PERIOD INPUT
default_period = 2

inp = input(f"Period [{default_period}]: ").strip()

try:
    PERIOD = int(inp) if inp else default_period
except ValueError:
    print("Input tidak valid, menggunakan default")
    PERIOD = default_period


# RTO TIME

default_rtotime = 5

inp = input(f"Lama RTO [{default_rtotime}]: ").strip()

try:
    RTO_TIME = int(inp) if inp else default_rtotime
except ValueError:
    print("Input tidak valid, menggunakan default")
    RTO_TIME = default_rtotime

# mapping label asli -> index model
label_to_index = {
    0:0,
    3:1,
    5:2,
    8:3,
    9:4
}
index_to_label = {
    0:0,
    1:3,
    2:5,
    3:8,
    4:9
}

# =============================
# SERIAL INIT
# =============================

print("Opening serial:", PORT)

ser = serial.Serial(PORT, BAUD, timeout=1)

time.sleep(2)
ser.reset_input_buffer()

# =============================
# HANDSHAKE
# =============================

def handshake():

    print("Checking ESP32 connection...")

    ser.write(b"PING\n")

    start = time.time()

    while time.time() - start < 5:

        raw = ser.readline()

        if raw == b"":
            continue

        line = raw.decode(errors="ignore").strip()

        print("ESP:", line)

        if line.startswith("PONG"):

            parts = line.split(",")

            if len(parts) >= 2:
                model_name = parts[1]
            else:
                model_name = "UNKNOWN_MODEL"

            print("ESP32 connection OK")
            print("Detected model:", model_name)

            return model_name

    return None


MODEL = handshake()

if MODEL is None:

    print("\nESP32 not responding")

    print("1. Retry handshake")
    print("2. Continue benchmark anyway")

    choice = input("Choose (1/2): ")

    if choice == "1":
        MODEL = handshake()

    elif choice == "2":
        MODEL = "UNKNOWN_MODEL"

# while not connected:

#     print("\nESP32 not responding")

#     print("1. Retry handshake")
#     print("2. Continue benchmark anyway")

#     choice = input("Choose (1/2): ")

#     if choice == "1":
#         connected = handshake()

#     elif choice == "2":
#         print("Continuing without handshake\n")
#         break


# =============================
# CSV LOGGER
# =============================

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
filename = f"results/{MODEL}_{timestamp}.csv"
eval_filename = f"results/{MODEL}_{timestamp}_eval.txt"

csv_file = open(filename, "w", newline="")
writer = csv.writer(csv_file)

writer.writerow([
    
    "test",
    "file",

    "gt_index",
    "gt_label",

    "pred_index",
    "pred_label",

    "correct",

    "latency_mcu_ms",
    "latency_pc_s",

    "tensor_arena",
    "free_ram",

    "iteration",
    "status",

    "timestamp"
])

print("Logging to:", filename)


# =============================
# DATASET
# =============================

def get_random_image():

    classes = [
        d for d in os.listdir(DATASET_PATH)
        if os.path.isdir(os.path.join(DATASET_PATH, d))
    ]

    cls = random.choice(classes)

    files = os.listdir(os.path.join(DATASET_PATH, cls))

    img_file = random.choice(files)

    path = os.path.join(DATASET_PATH, cls, img_file)

    img = np.fromfile(path, dtype=np.uint8)

    if len(img) != IMG_BYTES:
        raise ValueError(f"Invalid image size: {path}")

    # return img, int(cls), path
    gt_label = int(cls)
    gt_index = label_to_index[gt_label]

    return img, gt_index, path

# =============================
# SEND IMAGE
# =============================

def send_image(img):

    ser.write(b'\xAA')
    ser.write(img.tobytes())


# =============================
# BENCHMARK LOOP
# =============================

peak_ram = 0
latencies = []
crash_count = 0
wdt_count = 0

try:

    for i in range(NUM_TEST):

        iter_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        img, gt_class, path = get_random_image()

        start = time.time()

        send_image(img)

        pred = -1
        latency_mcu = None
        arena = None
        ram = None
        iteration = None
        status = "RTO"



        t0 = time.time()

        while True:

            raw = ser.readline()

            if raw == b"":
                if time.time() - t0 > RTO_TIME:
                    break
                continue

            # print("RAW:", raw)

            line = raw.decode(errors="ignore").strip()

            # print("ESP:", line)

            # print("timestamp:", iter_timestamp)

            if "Benchmark Ready" in line:
                status = "RESTART"
                print(
                        f"ITER:{i} | "
                        f"STATUS:{status} | "
                        f"TS:{iter_timestamp}"
                    )
                break

            if "PRED:" in line:

                try:

                    parts = line.split(",")

                    pred = int(parts[0].split(":")[1])
                    latency_mcu = int(parts[1].split(":")[1])
                    arena = int(parts[2].split(":")[1])
                    ram = int(parts[3].split(":")[1])
                    iteration = int(parts[4].split(":")[1])

                    status = "OK"

                    if latency_mcu is not None:
                        latencies.append(latency_mcu)

                    if ram is not None:
                        # peak_ram = min(peak_ram, ram) if peak_ram else ram
                        peak_ram = min(peak_ram, ram)
                    
                    print(
                        f"ITER:{i} | "
                        f"GT:{index_to_label[gt_class]} | "
                        f"PRED:{index_to_label.get(pred,-1)} | "
                        f"MCU_LAT:{latency_mcu}ms | "
                        f"RAM:{ram} | "
                        f"ARENA:{arena} | "
                        f"STATUS:{status} | "
                        f"TS:{iter_timestamp}"
                    )

                except Exception as e:

                    print("Parse error:", e)
                    status = "PARSE_ERR"

                break

            if "Guru Meditation" in line or "abort" in line:
                status = "CRASH"
                crash_count += 1
                print(
                        f"ITER:{i} | "
                        f"STATUS:{status} | "
                        f"TS:{iter_timestamp}"
                    )
                break

            if "WDT" in line or "watchdog" in line.lower():
                status = "WDT_RESET"
                wdt_count += 1
                print(
                        f"ITER:{i} | "
                        f"STATUS:{status} | "
                        f"TS:{iter_timestamp}"
                    )
                break
        
        # TAMBAHKAN INI
        if status == "RTO":
            print(
                f"ITER:{i} | "
                f"STATUS:{status} | "
                f"TS:{iter_timestamp}"
            )

        latency_pc = time.time() - start

        correct = int(pred == gt_class)

        writer.writerow([

            i,
            path,

            gt_class,                     # gt_index
            index_to_label[gt_class],     # gt_label

            pred,                         # pred_index
            index_to_label.get(pred,-1),  # pred_label

            correct,

            latency_mcu,
            latency_pc,

            arena,
            ram,

            iteration,
            status,

            iter_timestamp
        ])

        time.sleep(PERIOD)


except KeyboardInterrupt:

    print("\nBenchmark interrupted by user")

finally:

    csv_file.close()
    ser.close()

    print("\nBenchmark selesai")

# =============================
# EVALUATION
# =============================

eval_file = open(eval_filename, "w")

def log_eval(text):
    print(text)
    eval_file.write(text + "\n")

log_eval("\n===== BENCHMARK EVALUATION =====")

df = pd.read_csv(filename)

total = len(df)

ok = len(df[df["status"] == "OK"])
rto = len(df[df["status"] == "RTO"])
parse_err = len(df[df["status"] == "PARSE_ERR"])
restart = len(df[df["status"] == "RESTART"])

log_eval("\n===== SYSTEM RELIABILITY =====")

log_eval(f"Total Test        : {total}")
log_eval(f"Successful        : {ok}")
log_eval(f"Timeout (RTO)     : {rto}")
log_eval(f"Parse Error       : {parse_err}")
log_eval(f"ESP Restart       : {restart}")

log_eval(f"Crash Count       : {crash_count}")
log_eval(f"Watchdog Reset    : {wdt_count}")

success_rate = ok / total * 100 if total > 0 else 0

log_eval(f"Success Rate      : {success_rate:.2f}%")

if len(latencies) > 0:

    avg_latency = np.mean(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)

    fps = 1000 / avg_latency

    log_eval("\n===== PERFORMANCE =====")

    log_eval(f"Avg Latency (ms)  : {avg_latency:.2f}")
    log_eval(f"Min Latency (ms)  : {min_latency}")
    log_eval(f"Max Latency (ms)  : {max_latency}")
    std_latency = np.std(latencies)
    log_eval(f"Latency Std Dev   : {std_latency:.2f}")

    log_eval(f"FPS               : {fps:.2f}")

    log_eval(f"Peak Free RAM     : {peak_ram}")


# =============================
# MODEL ACCURACY
# =============================

df_ok = df[df["status"] == "OK"].copy()

if len(df_ok) > 0:

    latency_ok = df_ok["latency_mcu_ms"].dropna()

    if len(latency_ok) > 0:

        # log_eval("\n===== LATENCY ANALYSIS =====")

        # log_eval(f"Avg Latency      : {latency_ok.mean():.2f} ms")
        # log_eval(f"Median Latency   : {latency_ok.median():.2f} ms")
        # log_eval(f"Std Latency      : {latency_ok.std():.2f} ms")
        log_eval(f"P95 Latency      : {latency_ok.quantile(0.95):.2f} ms")

if len(df_ok) > 0:

    y_true = df_ok["gt_label"].astype(int)
    y_pred = df_ok["pred_label"].astype(int)

    correct = (y_true == y_pred).sum()
    total_ok = len(df_ok)

    accuracy = correct / total_ok * 100

    log_eval("\n===== MODEL ACCURACY =====")

    log_eval(f"Evaluated Samples : {total_ok}")
    log_eval(f"Correct           : {correct}")
    log_eval(f"Accuracy          : {accuracy:.2f}%")

    # =============================
    # CONFUSION MATRIX
    # =============================

    labels = sorted(y_true.unique())

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    log_eval("\n===== CONFUSION MATRIX =====")
    log_eval(str(cm))

    # =============================
    # CLASSIFICATION REPORT
    # =============================

    log_eval("\n===== CLASSIFICATION REPORT =====")

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0
    )

    log_eval(report)

else:

    log_eval("\nNo successful inference to evaluate")

eval_file.close()

# =============================
# SUMMARY REPORT (PER MODEL)
# =============================

summary_file = "results/benchmark_summary.csv"

summary_data = {

    "model": MODEL,

    "samples": total,
    "success": ok,
    "rto": rto,
    "crash": crash_count,

    "success_rate": success_rate,

    "accuracy": accuracy if len(df_ok) > 0 else None,

    "avg_latency_ms": avg_latency if len(latencies) > 0 else None,
    "p95_latency_ms": latency_ok.quantile(0.95) if len(df_ok) > 0 else None,

    "fps": fps if len(latencies) > 0 else None,

    "tensor_arena": arena,
    "peak_free_ram": peak_ram,

    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")

}

# =============================
# CREATE / APPEND SUMMARY CSV
# =============================

if not os.path.exists(summary_file):

    df_summary = pd.DataFrame([summary_data])
    df_summary.insert(0, "no", 1)

    df_summary.to_csv(summary_file, index=False)

    print("Summary CSV dibuat")

else:

    df_summary = pd.read_csv(summary_file)

    no_baru = len(df_summary) + 1
    summary_data["no"] = no_baru

    df_summary = pd.concat(
        [df_summary, pd.DataFrame([summary_data])],
        ignore_index=True
    )

    df_summary.to_csv(summary_file, index=False)

    print("Summary berhasil ditambahkan")