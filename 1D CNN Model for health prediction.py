# SIC AI CAPSTONE PROJECT
# Team: DataVisionaries :)
# Project: AI Health Status Predictor

import pandas as pd
import numpy as np
import datetime
import tkinter as tk
from tkinter import ttk, messagebox
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
import os
import time

# ---------------- Step 2: Generate realistic synthetic data ----------------
output_filename = "data.csv"

PATIENT_PROFILES = {
    'Healthy': {'temp': 36.8, 'sbp': 115, 'dbp': 75, 'hr': 70, 'std': {'temp':0.2,'sbp':5,'dbp':4,'hr':6}},
    'Fever': {'temp': 38.5, 'sbp': 125, 'dbp': 85, 'hr': 105, 'std': {'temp':0.3,'sbp':6,'dbp':5,'hr':8}},
    'Hypertension': {'temp': 36.8, 'sbp': 145, 'dbp': 95, 'hr': 75, 'std': {'temp':0.2,'sbp':7,'dbp':6,'hr':6}},
    'Hypotension': {'temp': 36.5, 'sbp': 85, 'dbp': 55, 'hr': 65, 'std': {'temp':0.2,'sbp':5,'dbp':4,'hr':5}},
    'Tachycardia': {'temp': 37.0, 'sbp': 120, 'dbp': 80, 'hr': 110, 'std': {'temp':0.2,'sbp':5,'dbp':4,'hr':7}},
    'Bradycardia': {'temp': 36.6, 'sbp': 110, 'dbp': 70, 'hr': 50, 'std': {'temp':0.2,'sbp':5,'dbp':4,'hr':5}},
    'Hypertension_with_Tachycardia': {'temp': 37.1, 'sbp': 150, 'dbp': 98, 'hr': 115, 'std': {'temp':0.3,'sbp':6,'dbp':5,'hr':8}},
    'Hypertension_with_Bradycardia': {'temp': 36.7, 'sbp': 148, 'dbp': 92, 'hr': 52, 'std': {'temp':0.2,'sbp':6,'dbp':5,'hr':5}},
    'Hypertension_with_Fever': {'temp': 38.2, 'sbp': 142, 'dbp': 93, 'hr': 98, 'std': {'temp':0.3,'sbp':6,'dbp':5,'hr':7}},
    'Hypotension_with_Tachycardia': {'temp': 36.6, 'sbp': 82, 'dbp': 52, 'hr': 112, 'std': {'temp':0.2,'sbp':5,'dbp':4,'hr':7}},
    'Hypotension_with_Bradycardia': {'temp': 36.4, 'sbp': 80, 'dbp': 50, 'hr': 48, 'std': {'temp':0.2,'sbp':5,'dbp':4,'hr':5}},
    'Hypotension_with_Fever': {'temp': 38.0, 'sbp': 88, 'dbp': 58, 'hr': 100, 'std': {'temp':0.3,'sbp':5,'dbp':4,'hr':7}},
    'Tachycardia_with_Fever': {'temp': 38.5, 'sbp': 125, 'dbp': 85, 'hr': 115, 'std': {'temp':0.3,'sbp':5,'dbp':4,'hr':8}},
    'Bradycardia_with_Hypothermia': {'temp': 34.8, 'sbp': 105, 'dbp': 65, 'hr': 45, 'std': {'temp':0.2,'sbp':5,'dbp':4,'hr':5}}
}

def generate_patient_data(patient_id, profile_name):
    profile = PATIENT_PROFILES[profile_name]
    num_readings = 48 * 6

    base_temp = np.random.normal(profile['temp'], profile['std']['temp'])
    base_sbp  = np.random.normal(profile['sbp'], profile['std']['sbp'])
    base_dbp  = np.random.normal(profile['dbp'], profile['std']['dbp'])
    base_hr   = np.random.normal(profile['hr'], profile['std']['hr'])

    time_hours = np.arange(num_readings) / 6
    circadian = np.sin(2 * np.pi * time_hours / 24)

    temps = np.random.normal(base_temp, 0.3, num_readings) + 0.2 * circadian
    sbps  = np.random.normal(base_sbp, 5, num_readings) + 1.5 * circadian
    dbps  = np.random.normal(base_dbp, 4, num_readings) + 1.0 * circadian
    hrs   = np.random.normal(base_hr, 6, num_readings) + 4 * circadian

    drift = np.linspace(0, np.random.normal(0, 0.5), num_readings)
    temps += 0.1 * drift
    sbps  += 0.2 * drift
    dbps  += 0.1 * drift
    hrs   += 0.3 * drift

    # Reduce extreme outliers
    for arr in [temps, sbps, dbps, hrs]:
        for i in range(num_readings):
            if np.random.rand() < 0.005:
                arr[i] += np.random.randint(-15, 15)

    start_time = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    timestamps = [start_time + datetime.timedelta(minutes=10*i) for i in range(num_readings)]

    patient_df = pd.DataFrame({
        'Patient_ID': patient_id,
        'Timestamp': timestamps,
        'Temperature_C': np.round(temps, 2),
        'Systolic_BP_mmHg': np.round(sbps).astype(int),
        'Diastolic_BP_mmHg': np.round(dbps).astype(int),
        'Heart_Rate_bpm': np.round(hrs).astype(int),
        'Health_Profile': profile_name
    })
    return patient_df

if not os.path.exists(output_filename):
    print("--- Dataset not found. Generating new synthetic data... ---")
    start_time_gen = time.time()
    total_patients = 15000
    all_patient_data = []
    profile_names = list(PATIENT_PROFILES.keys())
    for i in range(total_patients):
        random_profile_name = np.random.choice(profile_names)
        patient_df = generate_patient_data(i+1, random_profile_name)
        all_patient_data.append(patient_df)
        if (i+1) % 1000 == 0:
            print(f"  ...{i+1}/{total_patients} patients generated.")
    final_dataset = pd.concat(all_patient_data, ignore_index=True)
    final_dataset.to_csv(output_filename, index=False)
    print(f"Dataset generated and saved to '{output_filename}' in {time.time() - start_time_gen:.2f} seconds.")
else:
    print(f"--- Found existing dataset ('{output_filename}'). Loading data... ---")

# ---------------- Step 3: Load and preprocess ----------------
df = pd.read_csv(output_filename)
label_encoder = LabelEncoder()
df['Profile_Encoded'] = label_encoder.fit_transform(df['Health_Profile'])
class_names = list(label_encoder.classes_)
num_classes = len(class_names)
target_col = 'Profile_Encoded'
df = df.drop(columns=['Patient_ID','Timestamp','Health_Profile'])

feature_cols = ['Temperature_C','Systolic_BP_mmHg','Diastolic_BP_mmHg','Heart_Rate_bpm']
X = df[feature_cols].values
y = df[target_col].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

num_patients = len(df)//(48*6)
timesteps_per_patient = 48*6
num_features = len(feature_cols)
X_reshaped = X_scaled.reshape(num_patients, timesteps_per_patient, num_features)
y_reshaped = y[::timesteps_per_patient]

# ---------------- Step 4: Train-test split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y_reshaped, test_size=0.2, random_state=42, stratify=y_reshaped
)

# ---------------- Step 5: Build CNN Model ----------------
model = Sequential([
    tf.keras.Input(shape=(timesteps_per_patient,num_features)),
    Conv1D(64,6,activation='relu'), MaxPooling1D(2), Dropout(0.3),
    Conv1D(128,6,activation='relu'), MaxPooling1D(2), Dropout(0.3),
    Flatten(),
    Dense(128,activation='relu'),
    Dense(num_classes,activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ---------------- Step 6: Train Model ----------------
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=150, batch_size=64, validation_split=0.2, verbose=1, callbacks=[early_stop])

# ---------------- Step 7: Evaluate Model ----------------
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
per_class_report = classification_report(y_test, y_pred_classes, target_names=class_names, digits=3)

# ---------------- Step 8: UI ----------------
UI_PROFILES = {k.replace('_',' '):v for k,v in PATIENT_PROFILES.items()}

generated_data_for_prediction = None
def generate_vitals_from_profile():
    global generated_data_for_prediction
    profile_name = profile_combobox.get()
    if not profile_name:
        messagebox.showerror("Input Error","Please select a patient profile first.")
        return
    profile = UI_PROFILES[profile_name]
    num_readings = 24*6
    std_devs = {'temp':0.25,'sbp':7,'dbp':5,'hr':10}
    temps = np.random.normal(profile['temp'],std_devs['temp'],num_readings)
    sbps  = np.random.normal(profile['sbp'],std_devs['sbp'],num_readings)
    dbps  = np.random.normal(profile['dbp'],std_devs['dbp'],num_readings)
    hrs   = np.random.normal(profile['hr'],std_devs['hr'],num_readings)
    generated_data_for_prediction = np.array([temps,sbps,dbps,hrs]).T
    display_text = "Generated 24-Hour Vitals (144 readings):\n"
    display_text += "Time | Temp | SBP | DBP | HR\n"
    display_text += "---------------------------------\n"
    for i in range(num_readings):
        display_text += f"t-{143-i:03d} | {temps[i]:.1f} | {sbps[i]:.0f} | {dbps[i]:.0f} | {hrs[i]:.0f}\n"
    vitals_box.delete("1.0",tk.END)
    vitals_box.insert("1.0",display_text)
    predict_button.config(state="normal")

def run_prediction():
    if generated_data_for_prediction is None:
        messagebox.showerror("Error","Please generate patient data first.")
        return
    padding_needed = timesteps_per_patient - len(generated_data_for_prediction)
    padding = np.tile(generated_data_for_prediction[0],(padding_needed,1))
    full_sequence = np.vstack([padding, generated_data_for_prediction])
    scaled_sequence = scaler.transform(full_sequence)
    reshaped_sequence = scaled_sequence.reshape(1,timesteps_per_patient,num_features)
    probabilities = model.predict(reshaped_sequence)[0]
    results_with_names = list(zip(class_names, probabilities))
    sorted_results = sorted(results_with_names,key=lambda item:item[1],reverse=True)
    predicted_class_name = sorted_results[0][0]
    confidence = sorted_results[0][1]

    result_text = f"Final Prediction: {predicted_class_name}\nConfidence: {confidence:.2%}\n"
    if confidence < 0.75:
        result_text += "⚠️ Prediction is low confidence. Interpret results with caution.\n"
    result_text += "\nFull Probability Distribution:\n"
    for label, prob in sorted_results:
        result_text += f"{label.replace('_',' '):<35}: {prob:.2%}\n"

    result_box.delete("1.0",tk.END)
    result_box.insert("1.0",result_text)

# ---------------- UI ----------------
root = tk.Tk()
root.title("AI Health Status Predictor")
main_frame = ttk.Frame(root,padding=10)
main_frame.pack(fill="both",expand=True)

controls_frame = ttk.Frame(main_frame)
controls_frame.pack(fill="x",pady=5)
ttk.Label(controls_frame,text="Select Patient Profile:").pack(side="left",padx=5)
profile_combobox = ttk.Combobox(controls_frame,values=list(UI_PROFILES.keys()),width=30,state="readonly")
profile_combobox.pack(side="left",padx=5)
ttk.Button(controls_frame,text="Generate 24-Hour Vitals",command=generate_vitals_from_profile).pack(side="left",padx=10)

vitals_frame = ttk.LabelFrame(main_frame,text="Generated Vitals Sequence",padding=10)
vitals_frame.pack(fill="both",expand=True,pady=5)
vitals_box = tk.Text(vitals_frame,wrap="none",font=("Courier New",10),height=10)
vitals_scroll_y = ttk.Scrollbar(vitals_frame,orient="vertical",command=vitals_box.yview)
vitals_scroll_x = ttk.Scrollbar(vitals_frame,orient="horizontal",command=vitals_box.xview)
vitals_box.config(yscrollcommand=vitals_scroll_y.set,xscrollcommand=vitals_scroll_x.set)
vitals_scroll_y.pack(side="right",fill="y")
vitals_scroll_x.pack(side="bottom",fill="x")
vitals_box.pack(side="left",fill="both",expand=True)

predict_frame = ttk.Frame(main_frame)
predict_frame.pack(fill="x",pady=10)
predict_button = ttk.Button(predict_frame,text="Run Prediction on Generated Data",command=run_prediction,state="disabled")
predict_button.pack()

result_frame = ttk.LabelFrame(main_frame,text="Prediction Results",padding=10)
result_frame.pack(fill="both",expand=True,pady=5)
result_box = tk.Text(result_frame,wrap="word",font=("Courier New",11),height=10)
result_box.pack(fill="both",expand=True)

accuracy_frame = ttk.LabelFrame(main_frame, text="Per-Class Accuracy", padding=10)
accuracy_frame.pack(fill="both",expand=True,pady=5)
accuracy_box = tk.Text(accuracy_frame, wrap="word", font=("Courier New",11), height=10)
accuracy_box.pack(fill="both",expand=True)
accuracy_box.insert("1.0", f"Overall Test Accuracy: {accuracy*100:.2f}%\n\nPer-Class Report:\n{per_class_report}")

root.mainloop()
