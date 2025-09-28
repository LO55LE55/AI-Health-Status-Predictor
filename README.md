# AI Health Status Predictor  

An interactive AI tool that predicts patient health status from vital signs using deep learning — built to help clinicians catch potential health risks earlier.  

---

## 📖 Overview  
This project uses a **1D Convolutional Neural Network (CNN)** to analyze time-series data of patient vitals (temperature, heart rate, systolic and diastolic blood pressure).  
It’s trained on a **synthetic dataset of 15,000 patients**, each with 288 readings (one every 10 minutes for 48 hours) across **14 different health profiles**, including hypertension, tachycardia, bradycardia, and combinations like hypertension-with-fever.

The result is a model with **~93% accuracy**, designed to **prioritize recall** so that risky cases are less likely to be missed.  

---

## ✨ Key Features  
- 🧠 **Deep Learning Model:** 1D CNN optimized for time-series health data  
- 🗃 **Rich Synthetic Dataset:** Covers 14 health profiles, 288 readings per patient  
- ⚠️ **Confidence Alerts:** Flags low-confidence predictions for human review  
- 🎛 **Interactive Interface:** Simple Tkinter-based UI to generate vitals, run predictions, and view results  
- 📈 **Per-Class Performance Report:** Shows accuracy and classification report by condition  

---

## 🛠️ Tech Stack  
- **Python 3.9+**  
- **TensorFlow/Keras** – for 1D CNN model  
- **Pandas & NumPy** – for data generation and preprocessing  
- **Scikit-learn** – for preprocessing and classification reporting  
- **Tkinter** – for interactive desktop interface  

---
