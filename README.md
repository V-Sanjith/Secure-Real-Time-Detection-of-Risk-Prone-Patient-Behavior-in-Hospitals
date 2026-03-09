# 🏥 Secure Real-Time Detection of Risk-Prone Patient Behavior in Hospitals

> A real-time AI-powered hospital patient monitoring system that detects risk-prone behaviors using computer vision and deep learning, built with YOLOv8 and Streamlit.

---

## 📌 Overview

This system continuously monitors hospital rooms via video feeds to identify potentially dangerous patient behaviors — specifically when a **single patient is moving alone** without staff presence. It provides real-time alerts to help prevent falls, unauthorized movement, and other risk-prone incidents.

---

## 🚀 Features

- 🎯 **Real-Time Patient Detection** using YOLOv8 (You Only Look Once v8)
- 🧠 **Temporal Analysis** with sliding-window movement tracking
- 🚨 **Automated Alerting** when a single patient is detected moving alone
- 📊 **Performance Dashboard** with confusion matrices, ROC curves, and metric charts
- 🔬 **Ablation Study** support for tuning temporal window `k` and movement threshold `τ`
- 📥 **Exportable Reports** in HTML format
- 🖥️ **Interactive Streamlit UI** with live video stream processing
- 🔄 **Demo Mode** — runs simulated detection even without a connected camera or GPU

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| UI Framework | [Streamlit](https://streamlit.io/) |
| Object Detection | [YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics) |
| Deep Learning | [PyTorch](https://pytorch.org/) |
| Computer Vision | [OpenCV](https://opencv.org/) |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn, Plotly |
| Metrics | scikit-learn |

---

## 🧩 System Architecture

```
Video Input (Webcam / Uploaded File)
         ↓
   YOLOv8 Inference
   (Person Detection)
         ↓
  Temporal Analyzer
  (Sliding Window: k frames)
  (Movement Threshold: τ pixels)
         ↓
  Room Activity Analysis
  ┌─────────────────────────┐
  │ EMPTY → No Action       │
  │ SINGLE PATIENT IDLE → OK│
  │ SINGLE PATIENT MOVING   │
  │   → ⚠️ WARNING ALERT   │
  │ MULTIPLE PEOPLE → OK    │
  └─────────────────────────┘
         ↓
  Annotated Frame + Alert
```

---

## ⚙️ Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CONFIDENCE_THRESHOLD` | 0.5 | YOLO detection confidence cutoff |
| `IOU_THRESHOLD` | 0.5 | Intersection over Union for NMS |
| `TEMPORAL_WINDOW` (k) | 10 | Number of frames for movement averaging |
| `MOVEMENT_THRESHOLD` (τ) | 15.0 | Pixel displacement to classify as moving |

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/V-Sanjith/Secure-Real-Time-Detection-of-Risk-Prone-Patient-Behavior-in-Hospitals.git

cd Secure-Real-Time-Detection-of-Risk-Prone-Patient-Behavior-in-Hospitals

# Install dependencies
pip install streamlit ultralytics opencv-python torch numpy pandas matplotlib seaborn plotly scikit-learn pillow

# Run the app
streamlit run revisedhos.py
```

---

## 🖥️ Usage

1. Launch the app with `streamlit run revisedhos.py`
2. Choose between **Live Camera** or **Upload Video** mode
3. The system detects persons in each frame using YOLOv8
4. Movement is tracked over a sliding temporal window
5. A **WARNING** is triggered if a single patient is found moving without staff
6. Navigate to the **Results & Analysis** tab for detailed performance metrics

---

## 📊 Performance Metrics

The system is evaluated on:
- **Patient Detection** — Precision, Recall, F1-Score, Accuracy
- **Movement Classification** — Stationary vs. Moving accuracy
- **Alert Generation** — Alert Precision, Recall, F1-Score
- **ROC Curves** and **Confusion Matrices** for all tasks

---

## 🔬 Ablation Study

Configurable parameters tested:
- Temporal Windows: `[3, 5, 7]` frames
- Movement Thresholds: `[15.0, 25.0, 35.0]` pixels

---

## 👨‍💻 Team

**T77 — Secure Real-Time Detection of Risk-Prone Patient Behavior in Hospitals**  
B.Tech Final Year Project (2022–26 Batch)

---

## 📄 License

This project is for academic purposes.
