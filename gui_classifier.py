import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import joblib

# Load saved SVM model
model = joblib.load('svm_cats_dogs_model.pkl')

IMAGE_SIZE = (64, 64)  # Resize to same size used in training

def predict_image(file_path):
    img = cv2.imread(file_path)
    img_resized = cv2.resize(img, IMAGE_SIZE)
    img_flatten = img_resized.flatten().reshape(1, -1)
    prediction = model.predict(img_flatten)
    return "Dog" if prediction[0] == 1 else "Cat"

# GUI setup
root = tk.Tk()
root.title("Cat vs Dog Classifier")

def upload_and_predict():
    file_path = filedialog.askopenfilename()
    if file_path:
        result = predict_image(file_path)
        img = Image.open(file_path)
        img = img.resize((200, 200))
        img_tk = ImageTk.PhotoImage(img)
        panel.configure(image=img_tk)
        panel.image = img_tk
        result_label.config(text=f"Prediction: {result}")

# GUI Elements
upload_btn = tk.Button(root, text="Upload Image", command=upload_and_predict, font=("Arial", 14))
upload_btn.pack(pady=10)

panel = tk.Label(root)
panel.pack()

result_label = tk.Label(root, text="", font=("Arial", 16))
result_label.pack(pady=10)

root.mainloop()
