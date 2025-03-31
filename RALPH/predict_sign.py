import sys, os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Constants
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
CURRENT_DIR = os.path.dirname(__file__)

# Set the model path
MODEL_PATH = sys.argv[1] if len(sys.argv) == 2 else os.path.join(CURRENT_DIR,"best_model.h5")

if not os.path.exists(MODEL_PATH):
    sys.exit("Model not found. Please train the model first.")
    
# Load the trained model
model = load_model(MODEL_PATH)

# String representation of the gtsrb dataset categories
CATEGORIES = [
    "Speed limit (20km/h)",
    "Speed limit (30km/h)",
    "Speed limit (50km/h)",
    "Speed limit (60km/h)",
    "Speed limit (70km/h)",
    "Speed limit (80km/h)",
    "End of speed limit (80km/h)",
    "Speed limit (100km/h)",
    "Speed limit (120km/h)",
    "No passing",
    "No passing for vehicles over 3.5 metric tons",
    "Right-of-way at the next intersection",
    "Priority road",
    "Yield",
    "Stop",
    "No vehicles",
    "Vehicles over 3.5 metric tons prohibited",
    "No entry",
    "General caution",
    "Dangerous curve to the left",
    "Dangerous curve to the right",
    "Double curve",
    "Bumpy road",
    "Slippery road",
    "Road narrows on the right",
    "Road work",
    "Traffic signals",
    "Pedestrians",
    "Children crossing",
    "Bicycles crossing",
    "Beware of ice/snow",
    "Wild animals crossing",
    "End of all speed and passing limits",
    "Turn right ahead",
    "Turn left ahead",
    "Ahead only",
    "Go straight or right",
    "Go straight or left",
    "Keep right",
    "Keep left",
    "Roundabout mandatory",
    "End of no passing",
    "End of no passing by vehicles over 3.5 metric tons"
]

def predict_image(file_path):
    """
    Predict the category of the traffic sign in the given image file.
    """
    try:
        # Load and preprocess the image
        image = cv2.imread(file_path)
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        image = np.array(image) / 255.0  # Normalize pixel values
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Predict using the model
        predictions = model.predict(image)
        predicted_category = np.argmax(predictions)
        accuracy = np.max(predictions)

        return predicted_category, CATEGORIES[predicted_category], accuracy
    
    except Exception as e:
        messagebox.showerror("Error", f"Failed to predict image: {e}")
        return None

def open_file():
    """
    Open a file dialog to select an image and display the prediction.
    """
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.ppm;*.jpg;*.jpeg;*.png;*.bmp;*.tiff;*.webp")]
    )
    if not file_path:
        return

    try:
        # Display the selected image
        image = Image.open(file_path)
        max_size = (250, 250)
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.image = photo

        # Predict the traffic sign category
        number, sign, probability  = predict_image(file_path)
        if sign and probability:
            results = f"\nNumber: {number} Sign: {sign}, Accuracy: {probability:.2f}"
            result_label.config(text=results)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to open image: {e}")

# Create the tkinter interface
window = tk.Tk()
window.title("Traffic Sign Predictor")

# Set dimensions of the interface
window.geometry('400x400')


# Create and place widgets
frame = tk.Frame(window)
frame.pack(pady=20)

text_label = tk.Label(frame, text="Upload Your Image", font=("Arial", 12, "bold"))
text_label.pack()

button = tk.Button(frame, text="Upload", command=open_file)
button.pack(pady=10)

image_label = tk.Label(frame)
image_label.pack()

result_label = tk.Label(frame, text="Prediction: Nothing Yet", font=("Arial", 12))
result_label.pack(pady=10)

# Run the tkinter main loop
window.mainloop()