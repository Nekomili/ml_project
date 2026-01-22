import os
import glob
import logging
import random
from collections import Counter

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from ultralytics import YOLO

# Logging config
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("HybridPipeline")

# set device
if torch.cuda.is_available():
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(torch.cuda.current_device())
else:
    device = torch.device("cpu")

# part 1 - set torch_home to use yolo11n.pt for AMP check
os.environ["TORCH_HOME"] = "../YOLO"

# Part 2 - Train YOLO
def train_yolo():
    print("\nTraining YOLO model...")
    model = YOLO('../YOLO/yolo11s.pt')
    model.train(
        data='../YOLO/config/materials.yaml',
        optimizer='AdamW',
        epochs=100,
        imgsz=768,
        batch=16,
        device=0,
        workers=4,
        amp=True,
        save=True,
        val=True,
        plots=True,
        patience=20,
        cache='disk',
        project='../YOLO/runs/train',
        name='materials_test_direct',
        exist_ok=True,
        lr0=0.0007,
        momentum=0.85,
        weight_decay=0.0003,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        box=0.05,
        cls=0.5,
        dfl=1.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.8,
        perspective=0.002,
        fliplr=0.5,
        mosaic=0.2,
        mixup=0.1,
        copy_paste=0.05
    )
    print("YOLO training completed.")

# Part 3 - Crop Images
def generate_cropped_images(image_root, label_root, output_root):
    print(f"\nCropping images from: {image_root} → {output_root}")
    os.makedirs(output_root, exist_ok=True)
    image_paths = sorted(glob.glob(os.path.join(image_root, '*.jpg')) + glob.glob(os.path.join(image_root, '*.png')))
    for img_path in image_paths:
        name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_root, f"{name}.txt")
        if not os.path.exists(label_path): continue
        img = cv2.imread(img_path)
        if img is None: continue
        h, w = img.shape[:2]
        with open(label_path, 'r') as f:
            for idx, line in enumerate(f):
                class_id, xc, yc, bw, bh = map(float, line.strip().split())
                x1 = max(0, int((xc - bw / 2) * w))
                y1 = max(0, int((yc - bh / 2) * h))
                x2 = min(w, int((xc + bw / 2) * w))
                y2 = min(h, int((yc + bh / 2) * h))
                cropped = img[y1:y2, x1:x2]
                if cropped.size == 0: continue
                out_path = os.path.join(output_root, str(int(class_id)), f"{name}_{idx}.jpg")
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                cv2.imwrite(out_path, cropped)
    print("Cropping completed.")

# Part 4 - Train DenseNet
def train_densenet():
    print("\nTraining DenseNet 121 model...")
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    train_data = datasets.ImageFolder('../YOLO/datasets/materials/cropped/train', transform=transform)
    val_data = datasets.ImageFolder('../YOLO/datasets/materials/cropped/validation', transform=transform)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    model.classifier = nn.Linear(model.classifier.in_features, len(train_data.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-4)
    best_val_acc = 0.0

    for epoch in range(25):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        print(f"Epoch {epoch+1:02d}: Train Acc = {train_acc:.2f}%, Val Acc = {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), '../YOLO/densenet_best.pth')

    print("DenseNet training complete.")

# Part 5 - Evaluate DenseNet
def evaluate_densenet():
    print("\n🔍 Evaluating DenseNet model...")
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    val_data = datasets.ImageFolder('../YOLO/datasets/materials/cropped/validation', transform=transform)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, len(val_data.classes))
    model.load_state_dict(torch.load("../YOLO/densenet_best.pth", map_location=device))
    model.to(device).eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print("\nDenseNet accuracy validation:")
    print(classification_report(y_true, y_pred, target_names=val_data.classes, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

# Part 6 - Evaluate Hybrid Model
def evaluate_hybrid():
    print("\nEvaluating hybrid model...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    class_names = datasets.ImageFolder("../YOLO/datasets/materials/cropped/train", transform=transform).classes
    densenet_model = models.densenet121(weights=None)
    densenet_model.classifier = nn.Linear(densenet_model.classifier.in_features, len(class_names))
    densenet_model.load_state_dict(torch.load("../YOLO/densenet_best.pth", map_location=device))
    densenet_model.to(device)
    densenet_model.eval()
    print("\n Selecting best run from YOLO model training.")
    yolo_model = YOLO("../YOLO/runs/train/materials_test_direct/weights/best.pt")

    image_paths = glob.glob("../YOLO/datasets/materials/images/validation/*.jpg") + \
                  glob.glob("../YOLO/datasets/materials/images/validation/*.png")

    results_counter = Counter()

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None: continue
        results = yolo_model(img_path)[0]
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            yolo_cls = int(box.cls[0])
            yolo_conf = float(box.conf[0])
            crop = img[y1:y2, x1:x2]
            if crop.shape[0] == 0 or crop.shape[1] == 0:
                continue
            pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            input_tensor = transform(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                output = densenet_model(input_tensor)
                densenet_cls = output.argmax(dim=1).item()
                densenet_conf = torch.softmax(output, dim=1)[0][densenet_cls].item()

            if yolo_conf >= 0.75:
                final_cls = yolo_cls
                source = "YOLO"
            else:
                final_cls = densenet_cls
                source = "DenseNet"

            results_counter[(class_names[final_cls], source)] += 1

    print("\nHybrid classification results:")
    for (cls_name, source), count in sorted(results_counter.items()):
        print(f"{cls_name:25s} [{source}]: {count} kpl")
    print("Hybrid model evaluation done.")

# Part 7 - Display Comparison
def show_yolo_vs_densenet(image_path):
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    class_names = datasets.ImageFolder("../YOLO/datasets/materials/cropped/train", transform=transform).classes

    yolo_model = YOLO("../YOLO/runs/train/materials_test_direct/weights/best.pt")
    densenet_model = models.densenet121(weights=None)
    densenet_model.classifier = nn.Linear(densenet_model.classifier.in_features, len(class_names))
    densenet_model.load_state_dict(torch.load("../YOLO/densenet_best.pth", map_location=device))
    densenet_model.to(device).eval()

    img = cv2.imread(image_path)
    yolo_img, densenet_img, hybrid_img = img.copy(), img.copy(), img.copy()

    results = yolo_model.predict(image_path, verbose=False)[0]
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        yolo_cls = int(box.cls[0])
        yolo_conf = float(box.conf[0])
        crop = densenet_img[y1:y2, x1:x2]
        if crop.size == 0: continue

        label = f"{class_names[yolo_cls]} ({yolo_conf*100:.1f}%)"
        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = densenet_model(input_tensor)
            pred_cls = output.argmax(dim=1).item()
            dn_conf = torch.softmax(output, dim=1)[0][pred_cls].item()

        hybrid_cls = yolo_cls if yolo_conf >= 0.75 else pred_cls
        hybrid_conf = yolo_conf if yolo_conf >= 0.75 else dn_conf
        source = "YOLO" if yolo_conf >= 0.75 else "DenseNet"

        font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4
        cv2.rectangle(hybrid_img, (x1, y1), (x2, y2), (0, 255, 255) if source == "YOLO" else (0, 255, 0), 2)
        cv2.putText(hybrid_img, f"{class_names[hybrid_cls]} ({hybrid_conf*100:.1f}%) [{source}]", (x1, max(30, y1 - 10)),
                    font, scale, (0, 255, 255) if source == "YOLO" else (0, 255, 0), thickness)

    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(hybrid_img, cv2.COLOR_BGR2RGB))
    plt.title("Hybrid Classification Result")
    plt.axis("off")
    plt.show()

# Part 8 - Run Random Examples
def run_random_examples():
    print("\nRunning 3 random examples with hybrid model...")
    all_images = glob.glob("../YOLO/datasets/materials/images/validation/*.jpg") + \
                 glob.glob("../YOLO/datasets/materials/images/validation/*.png")
    sample = random.sample(all_images, 3)
    for path in sample:
        show_yolo_vs_densenet(path)

# YOLo summary

def yolo_eval(weights_path):
    print("\nRetrieving YOLO data summary from best run...")
    model = YOLO(weights_path)
    results = model.val()

# ---- MAIN EXECUTION ----

if __name__ == "__main__":  
    train_yolo()
#    generate_cropped_images('../YOLO/datasets/materials/images/train', '../YOLO/datasets/materials/labels/train', '../YOLO/datasets/materials/cropped/train')
#    generate_cropped_images('../YOLO/datasets/materials/images/validation', '../YOLO/datasets/materials/labels/validation', '../YOLO/datasets/materials/cropped/validation')
    train_densenet()
    evaluate_hybrid()
    yolo_eval("../YOLO/runs/train/materials_test_direct/weights/best.pt")
    evaluate_densenet()
