import cv2
import numpy as np
from ultralytics import YOLO
import os
import torch

# KONFIGURASI PATH

MODEL_PATH = 'D:/KCBUAS/runs/detect/fold_0/weights/best.pt'
VIDEO_SOURCE_PATH = 'D:/KCBUAS/vidsample/10.mp4'
VIDEO_OUTPUT_PATH = 'D:/KCBUAS/vidhasil/3.10.mp4'

os.makedirs(os.path.dirname(VIDEO_OUTPUT_PATH), exist_ok=True)

tracker_path = "ultralytics/cfg/trackers/bytetrack.yaml"

# PERSPECTIVE TRANSFORMATION

image_pts = np.array([
    [735, 243],  # A
    [928, 243],  # B
    [1477, 912], # D 
    [1, 912]     # C

], dtype=np.float32)

LEBAR_JALAN_METER = 11
PANJANG_JALAN_METER = 40

world_pts = np.array([
    [0, 0],
    [LEBAR_JALAN_METER, 0],
    [LEBAR_JALAN_METER, PANJANG_JALAN_METER],
    [0, PANJANG_JALAN_METER]
], dtype=np.float32)

M = cv2.getPerspectiveTransform(image_pts, world_pts)
print("Matriks Transformasi (M) berhasil dibuat.")


# LOAD YOLO MODEL

try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(MODEL_PATH)
    print(f"Model berhasil dimuat: {MODEL_PATH} (device: {device})")
except Exception as e:
    print("Error saat memuat model:", e)
    raise SystemExit


# LOAD VIDEO

cap = cv2.VideoCapture(VIDEO_SOURCE_PATH)
if not cap.isOpened():
    print(f"Error: Tidak dapat membuka video di {VIDEO_SOURCE_PATH}")
    raise SystemExit

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, fps, (w, h))


# TRACKING + SPEED ESTIMATION

tracked_objects = {}  # track_id : (prev_x, prev_y, prev_time)
speed_history = {}    # track_id : list of previous speeds
ema_speed = {}        # track_id : last filtered speed

def classify_risk(speed_kmh):
    if speed_kmh >= 80: return "Tinggi", (0, 0, 255)
    elif speed_kmh >= 40: return "Sedang", (0, 255, 255)
    elif speed_kmh >= 20: return "Rendah", (0, 255, 0)
    else: return "Aman", (255, 255, 255)

print("Memulai proses prediksi dan perhitungan kecepatan...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video selesai diproses.")
        break

    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    # YOLO TRACK #
    results = model.track(frame, persist=True, tracker=tracker_path)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()
        clss = results[0].boxes.cls.cpu().numpy().astype(int)

        # PROSES SETIAP OBJEK #
        for box, track_id, conf, cls_id in zip(boxes, ids, confs, clss):
            x1, y1, x2, y2 = box
            center_x, center_y = int((x1 + x2) / 2), y2
            speed_kmh = 0

            #  HITUNG KECEPATAN  #
            if track_id in tracked_objects:
                prev_x, prev_y, prev_time = tracked_objects[track_id]
                selisih_waktu = current_time - prev_time
                if selisih_waktu > 0:
                    titik_lama = np.array([[[prev_x, prev_y]]], dtype=np.float32)
                    titik_baru = np.array([[[center_x, center_y]]], dtype=np.float32)
                    world_old = cv2.perspectiveTransform(titik_lama, M)
                    world_new = cv2.perspectiveTransform(titik_baru, M)
                    jarak_meter = np.linalg.norm(world_new - world_old)
                    speed_ms = jarak_meter / selisih_waktu
                    speed_kmh = speed_ms * 3.6

            #  UPDATE POSISI  #
            tracked_objects[track_id] = (center_x, center_y, current_time)

            #  STABILISASI KECEPATAN  #
            if speed_kmh > 200:
                speed_kmh = 200

            if track_id not in speed_history:
                speed_history[track_id] = []
            speed_history[track_id].append(speed_kmh)
            if len(speed_history[track_id]) > 5:
                speed_history[track_id].pop(0)
            avg_speed = sum(speed_history[track_id]) / len(speed_history[track_id])

            alpha = 0.25
            if track_id not in ema_speed:
                ema_speed[track_id] = avg_speed
            else:
                ema_speed[track_id] = (alpha * avg_speed) + (1 - alpha) * ema_speed[track_id]

            final_speed = ema_speed[track_id]

            #  RISK LABEL  #
            risk_label, color = classify_risk(final_speed)
            class_name = model.names[cls_id] if cls_id < len(model.names) else "Unknown"
            label1 = f"ID:{track_id} {class_name} {conf:.2f}"
            label2 = f"{final_speed:.2f} km/h - Risiko: {risk_label}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label1, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, label2, (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out.write(frame)


# SELESAI

cap.release()
out.release()
print("Proses selesai.")
print("Video tersimpan di:", VIDEO_OUTPUT_PATH)