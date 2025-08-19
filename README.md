# 🛡️ Face Anti-Spoofing with YOLO & OpenCV  

A real-time **face anti-spoofing system** built using **YOLO, OpenCV, and cvzone**.  
This project demonstrates how to collect face datasets, split them for training/validation, and run **spoof detection (fake vs. real)** directly from your webcam.  

 Uses YOLO for classification + bounding box detection  
 Dataset auto-split into train/val/test with labels  
 Real-time demo with green (real) and red (fake) bounding boxes  

---

##  Description  

This project simulates a **face anti-spoofing pipeline** where you can:  

- Collect **labeled face images** (fake vs. real) from a webcam.  
- Automatically split the dataset into **train/val/test sets**.  
- Use **YOLOv8** for training and real-time inference.  
- Detect and classify faces as **REAL (green)** or **FAKE (red)** in a live video feed.  

It’s a great mini-project to showcase **computer vision, dataset creation, and deep learning deployment** concepts.  

---

## ✨ Features  

-  **Webcam-based Data Collection** – captures and labels faces.  
-  **Automatic Label Generation** – YOLO-compatible `.txt` files.  
-  **Dataset Splitter** – creates train/val/test directories with labels.  
-  **YOLOv8 Integration** – for fast and accurate inference.  
-  **Real-Time Classification** – color-coded bounding boxes for clarity.    

---

## 🛠️ How to Use  

### 🔌 Hardware/Software Setup  
- A PC/laptop with webcam  
- Python 3.8+  
- Install dependencies:  
pip install opencv-python cvzone ultralytics

### Workflow 

## Collect Dataset
python datacollection.py

Captures faces from webcam
Saves .jpg images + .txt YOLO labels in Dataset/all/

### Split Dataset

**Automatically creates:**
- train/ → 70% data
- val/ → 20% data
- test/ → 10% data

Generates data.yaml file

**Run YOLO Testing (Pretrained Model)**
python train.py


---

## Classes  

| ID     | Label|
|--------|------|
| 0      | Fake |
| 1      | Real |

**Example Output**
- [REAL 95%]  
- [FAKE 87%]  
- FPS: 28

---

**Future Ideas**

- Expand dataset with more spoof types
- Integrate in meeting apps like Teams, Zoom

---

## 👩‍💻 Author

_Made with ❤️ by Mansi Wanjale
B.Tech CSE @ Cummins College of Engineering, Pune_

---
## License
MIT License – free to use, fork & build upon.
