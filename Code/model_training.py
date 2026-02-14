import os
import yaml
import shutil
import numpy as np
from glob import glob
from ultralytics import YOLO
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns


# 0. Folder untuk simpan grafik

GRAPH_DIR = "runs/metrics"
os.makedirs(GRAPH_DIR, exist_ok=True)


# 1. Copy otomatis data.yaml + copy isi train/valid ke semua fold

def prepare_folds():
    src_yaml = r"D:/KCBUAS/dataset/data.yaml"
    src_train = r"D:/KCBUAS/dataset/train"
    src_valid = r"D:/KCBUAS/dataset/valid"

    print("=== COPY DATASET KE SETIAP FOLD ===")

    for fold in range(5):
        fold_path = f"D:/KCBUAS/kfold/fold{fold}"

        # buat folder
        train_img = f"{fold_path}/train/images"
        train_lbl = f"{fold_path}/train/labels"
        valid_img = f"{fold_path}/valid/images"
        valid_lbl = f"{fold_path}/valid/labels"

        os.makedirs(train_img, exist_ok=True)
        os.makedirs(train_lbl, exist_ok=True)
        os.makedirs(valid_img, exist_ok=True)
        os.makedirs(valid_lbl, exist_ok=True)

        # copy yaml
        shutil.copy(src_yaml, f"{fold_path}/data.yaml")
        print(f"[COPY] data.yaml → fold{fold}")

        # COPY TRAIN ================
        print(f"[COPY] train → fold{fold}")
        for img in glob(src_train + "/images/*"):
            shutil.copy(img, train_img)
        for lbl in glob(src_train + "/labels/*"):
            shutil.copy(lbl, train_lbl)

        # COPY VALID =================
        print(f"[COPY] valid → fold{fold}")
        for img in glob(src_valid + "/images/*"):
            shutil.copy(img, valid_img)
        for lbl in glob(src_valid + "/labels/*"):
            shutil.copy(lbl, valid_lbl)


# 2. Load class dari yaml

def load_classes(yaml_path):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    classes = data["names"]

    print("\n===== KELAS YANG DILATIH =====")
    for i, cls in enumerate(classes):
        print(f"ID {i} = {cls}")
    print("==============\n")

    return classes


# 3. Train model untuk satu fold (aman untuk VRAM 4GB)

def train_fold(fold):
    print(f"\n[TRAIN] Fold {fold}\n")
    model = YOLO("yolo11m.pt")  # pretrained

    fold_dir = f"D:/KCBUAS/kfold/fold{fold}"
    yaml_path = os.path.join(fold_dir, "data.yaml")

    load_classes(yaml_path)

    #  TRAIN 
    model.train(
        data=yaml_path,
        epochs=100,          # target epoch
        batch=2,             # batch lebih kecil agar aman VRAM 4GB
        imgsz=416,           # ukuran gambar lebih kecil
        name=f"fold_{fold}",
        pretrained=True,
        workers=2,
        half=True,           # gunakan FP16 untuk hemat VRAM
        device="cuda:0",     
        resume=False          # lanjutkan dari last.pt jika ada
    )

    return model


# 4. Evaluasi tiap fold

def evaluate_fold(model, fold):
    print(f"[EVAL] Fold {fold}")
    img_dir = f"D:/KCBUAS/kfold/fold{fold}/valid/images"

    imgs = glob(img_dir + "/*.jpg") + glob(img_dir + "/*.png")

    y_true, y_pred = []

    for img in imgs:
        lbl_path = img.replace("images", "labels").replace(".jpg", ".txt").replace(".png", ".txt")
        if not os.path.exists(lbl_path):
            continue
        with open(lbl_path, "r") as f:
            first_line = f.readline().strip().split()
            if len(first_line) == 0:
                continue
            y_true.append(int(first_line[0]))

        results = model.predict(img, conf=0.25, verbose=False)
        if len(results[0].boxes) == 0:
            y_pred.append(-1)
        else:
            confs = results[0].boxes.conf.cpu().numpy()
            clss = results[0].boxes.cls.cpu().numpy()
            y_pred.append(int(clss[np.argmax(confs)]))

    return y_true, y_pred


# 5. Plot Confusion Matrix

def plot_confusion(cm, labels, fold):
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix Fold {fold}")
    plt.xlabel("Prediksi")
    plt.ylabel("Label")
    plt.tight_layout()

    path = f"{GRAPH_DIR}/confusion_fold_{fold}.png"
    plt.savefig(path)
    plt.close()
    print(f"[SAVE] {path}")


# 6. Plot akurasi

def plot_accuracy(accs):
    plt.figure()
    plt.plot(range(5), accs, marker="o")
    plt.title("Akurasi Setiap Fold")
    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.tight_layout()

    path = f"{GRAPH_DIR}/accuracy_plot.png"
    plt.savefig(path)
    plt.close()
    print(f"[SAVE] {path}")


# 7. Plot mAP

def plot_map(maps50, maps95):
    plt.figure()
    plt.plot(range(5), maps50, marker="o", label="mAP50")
    plt.plot(range(5), maps95, marker="o", label="mAP50-95")
    plt.legend()
    plt.title("mAP Setiap Fold")
    plt.xlabel("Fold")
    plt.ylabel("mAP")
    plt.grid()
    plt.tight_layout()

    path = f"{GRAPH_DIR}/map_plot.png"
    plt.savefig(path)
    plt.close()
    print(f"[SAVE] {path}")


# 8. RUN K-FOLD

def main():
    prepare_folds()

    kelas = load_classes("D:/KCBUAS/kfold/fold0/data.yaml")

    accuracy_all = []
    map50_all = []
    map95_all = []

    for fold in range(5):
        print(f"\n=========== FOLD {fold} ===========")
        model = train_fold(fold)

        metrics = model.val()
        map50_all.append(metrics.results_dict["metrics/mAP50"])
        map95_all.append(metrics.results_dict["metrics/mAP50-95"])

        y_true, y_pred = evaluate_fold(model, fold)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(kelas))))
        acc = accuracy_score(y_true, y_pred)

        accuracy_all.append(acc)
        plot_confusion(cm, kelas, fold)

        print(f"[HASIL FOLD {fold}]")
        print("Confusion Matrix:\n", cm)
        print(f"Akurasi = {acc:.4f}")

    plot_accuracy(accuracy_all)
    plot_map(map50_all, map95_all)

    print("\n=========== RINGKASAN AKHIR ===========")
    print(f"Rata-rata Akurasi  : {np.mean(accuracy_all):.4f}")
    print(f"Rata-rata mAP50    : {np.mean(map50_all):.4f}")
    print(f"Rata-rata mAP50-95 : {np.mean(map95_all):.4f}")
    print("Semua grafik disimpan di folder: runs/metrics/")
    print("===\n")

if __name__ == "__main__":
    main()