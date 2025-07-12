# 🏗️ Real-Time Construction Site Safety Monitoring System

This project presents an **AI-powered system** for real-time monitoring of **Personal Protective Equipment (PPE)** compliance on construction sites. It leverages the state-of-the-art **YOLOv8** object detection model to identify workers and detect essential safety gear like **helmets** and **safety vests** from live or recorded video streams.

> ⚠️ The goal is to enhance worker safety by replacing manual supervision with an **automated, scalable, and consistent** solution.

---
## Sample Output
![GIF](output_ppe.gif)
---

---

## 🚀 Key Features

- 🎯 **Multi-Class Object Detection**  
  Detects and localizes:
  - 👷 `Person`
  - 🪖 `Helmet`
  - 🦺 `Safety Vest`

- 🧠 **Intelligent PPE Compliance Check**  
  Analyzes the spatial proximity of detected PPE to determine if a person is compliant.

- 🚨 **Real-Time Alerts**  
  Non-compliant individuals are flagged in **red** with an on-screen warning for immediate action.

---

## 🧰 Tech Stack

| Component     | Technology       |
|---------------|------------------|
| Model         | YOLOv8           |
| Framework     | PyTorch          |
| Libraries     | OpenCV, Ultralytics |
| Dataset       | Roboflow PPE Dataset |

---

## 📊 Performance Metrics

The model achieved a 99% accuracy in vehicle counting on the test video.
