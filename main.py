import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ============================
# STEP 1: LOAD AND CLEAN DATA
# ============================
data = pd.read_csv("loan.csv")
data.dropna(inplace=True)

# Drop Loan_ID if present
if "Loan_ID" in data.columns:
    data.drop("Loan_ID", axis=1, inplace=True)

target_col = "Loan_Status"
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

# Encode categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

X = data.drop(target_col, axis=1)
y = data[target_col]

# ============================
# STEP 2: TRAIN MODEL
# ============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
acc = accuracy_score(y_test, model.predict(X_test))
print(f"✅ Model trained with accuracy: {acc:.2f}")

# ============================
# STEP 3: BUILD MAIN WINDOW
# ============================
root = tk.Tk()
root.title("🏦 Loan Approval Predictor")
root.geometry("540x720")

header = tk.Frame(root)
header.pack(pady=10)

tk.Label(header, text="Loan Approval Predictor", font=("Arial", 18, "bold")).pack()
tk.Label(header, text=f"Model Accuracy: {acc:.2f}", font=("Arial", 12)).pack(pady=5)

# ============================
# STEP 4: SCROLLABLE INPUT AREA
# ============================
container = tk.Frame(root)
container.pack(fill="both", expand=True, padx=10, pady=5)

canvas = tk.Canvas(container)
scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# ============================
# STEP 5: CREATE INPUT FIELDS
# ============================
widgets = {}

for col in X.columns:
    tk.Label(scrollable_frame, text=col, font=("Arial", 11)).pack(pady=(5, 0))
    if col in label_encoders:
        combo = ttk.Combobox(scrollable_frame, values=list(label_encoders[col].classes_),
                             state="readonly", width=25)
        combo.current(0)
        combo.pack(pady=2)
        widgets[col] = combo
    else:
        entry = tk.Entry(scrollable_frame, width=28)
        entry.pack(pady=2)
        widgets[col] = entry

# ============================
# STEP 6: PREDICTION FUNCTION
# ============================
def predict():
    try:
        input_data = []
        for col in X.columns:
            if col in label_encoders:
                val = widgets[col].get()
                val_encoded = label_encoders[col].transform([val])[0]
                input_data.append(val_encoded)
            else:
                val = widgets[col].get()
                if val == "":
                    messagebox.showerror("Input Error", f"Please enter {col}")
                    return
                input_data.append(float(val))

        result = model.predict([input_data])[0]
        msg = "✅ Loan Approved" if result == 1 else "❌ Loan Not Approved"
        messagebox.showinfo("Prediction Result", msg)
    except Exception as e:
        messagebox.showerror("Error", str(e))

# ============================
# STEP 7: FIXED BUTTON AT BOTTOM
# ============================
button_frame = tk.Frame(root, bg="#f0f0f0", relief="raised", bd=1)
button_frame.pack(side="bottom", fill="x")

tk.Button(
    button_frame,
    text="Predict Loan Status",
    command=predict,
    bg="green",
    fg="white",
    font=("Arial", 13, "bold"),
    width=25,
    height=2
).pack(pady=10)

# Enable mousewheel scrolling
def _on_mousewheel(event):
    canvas.yview_scroll(-1 * int(event.delta / 120), "units")

canvas.bind_all("<MouseWheel>", _on_mousewheel)

root.mainloop()
