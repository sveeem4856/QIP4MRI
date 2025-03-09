import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import qiskit
from qiskit import QuantumCircuit, transpile, assemble
from qiskit_aer import Aer
import pennylane as qml
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, redirect, url_for, send_file




# Flask App Configuration
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"  # Folder to store images


# Ensure upload folder exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


# Load the pre-trained model
model = load_model('tumor_classifier.h5')




# Quantum Image Processing Functions
def quantum_process_image(img_path):
    """
    Process an image using quantum computing techniques
    """
    # Read the image
    img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_arr = np.array(img_array)
    height, width = img_arr.shape
    
    # Edge detection
    edges = cv2.Canny(img_arr, threshold1=100, threshold2=200)
    edges_float = edges.astype(np.float32) / 255.0
    
    # Blending the edges with the original image
    alpha = 0.9  # weight for the original image
    beta = 0.1   # weight for the edge mask
    
    img_float = img_arr.astype(np.float32)
    edges_float_scaled = edges_float * 255
    
    # Apply the blending
    enhanced = cv2.addWeighted(img_float, alpha, edges_float_scaled, beta, 0)
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    
    # Fourier Transform (2D FFT)
    f_transform = np.fft.fftshift(np.fft.fft2(enhanced))
    magnitude_spectrum = np.abs(f_transform)
    threshold = np.percentile(magnitude_spectrum, 99)
    f_transform_cleaned = np.where(np.abs(f_transform) > threshold, 0, f_transform)
    img_denoised = np.fft.ifft2(np.fft.ifftshift(f_transform_cleaned)).real
    img_denoised = np.clip(img_denoised, 0, 255).astype(np.uint8)
    
    # Quantum processing
    flattened_img = enhanced.flatten()
    norm_before = np.linalg.norm(flattened_img)
    quantized_pre_normalized = flattened_img.astype(np.float32)
    flattened = quantized_pre_normalized / 255.0
    
    # Prepare for quantum processing
    num_qubits = int(np.ceil(np.log2(len(flattened))))
    num_pixels = 2**num_qubits
    
    # Resize the flattened array to fit 2^n elements
    if len(flattened_img) < num_pixels:
        padded_img = np.zeros(num_pixels)
        padded_img[: len(flattened_img)] = flattened_img
    else:
        padded_img = flattened_img[:num_pixels]
    
    # Normalize the pixel values to ensure sum of squares = 1
    padded_img = padded_img / np.linalg.norm(padded_img)
    
    # Set up quantum device
    dev = qml.device("default.qubit", wires=num_qubits)
    
    @qml.qnode(dev)
    def quantum_encoding_amplification(f, num_iterations):
        # Amplitude Embedding of the image data
        qml.AmplitudeEmbedding(features=f, wires=range(num_qubits), normalize=True)
        
        # Apply Grover iterations
        for _ in range(num_iterations):
            # Oracle
            qml.PauliX(wires=num_qubits - 1)
            qml.Hadamard(wires=num_qubits - 1)
            qml.MultiControlledX(control_wires=list(range(num_qubits - 1)), wires=num_qubits - 1)
            qml.Hadamard(wires=num_qubits - 1)
            qml.PauliX(wires=num_qubits - 1)
            
            # Diffusion Operator
            for w in range(num_qubits):
                qml.Hadamard(wires=w)
            for w in range(num_qubits):
                qml.PauliX(wires=w)
            qml.Hadamard(wires=num_qubits - 1)
            qml.MultiControlledX(control_wires=list(range(num_qubits - 1)), wires=num_qubits - 1)
            qml.Hadamard(wires=num_qubits - 1)
            for w in range(num_qubits):
                qml.PauliX(wires=w)
            for w in range(num_qubits):
                qml.Hadamard(wires=w)
        
        return qml.state()
    
    # Calculate optimal number of iterations
    optimal_iterations = int(np.floor((np.pi / 4) * np.sqrt(num_qubits)))
    
    # Run quantum circuit
    quantum_state = quantum_encoding_amplification(padded_img, optimal_iterations + 1)
    
    # Process quantum output
    probabilities = np.abs(quantum_state) ** 2
    norm_after = np.linalg.norm(probabilities)
    probabilities = probabilities * (norm_before/norm_after)
    scaled_probabilities = np.clip(probabilities, 0, 255)
    
    # Reshape to original image dimensions
    if len(scaled_probabilities) < height * width:
        scaled_probabilities = np.pad(scaled_probabilities, (0, height * width - len(scaled_probabilities)), 'constant')
    elif len(scaled_probabilities) > height * width:
        scaled_probabilities = scaled_probabilities[:height * width]
    
    enhanced_image_data = scaled_probabilities.reshape((height, width))
    
    # Save the enhanced image
    output_path = os.path.join(app.config["UPLOAD_FOLDER"], f"quantum-{os.path.basename(img_path)}")
    enhanced_img = Image.fromarray(enhanced_image_data.astype(np.uint8))
    enhanced_img.save(output_path)
    
    return output_path




def predict_image(image_path):
    """
    Predict if an image contains a tumor
    """
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0  # Rescale pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension


    # Make prediction
    prediction = model.predict(img_array)
    result = "Tumor" if prediction > 0.5 else "No Tumor"
    confidence = prediction[0][0] * 100 if result == "Tumor" else (1 - prediction[0][0]) * 100


    return result, confidence




# Flask routes
@app.route("/", methods=["GET", "POST"])
def main():
    if request.method == "POST":
        q1 = request.form.get("q1")
        q2 = request.form.get("q2")
        q3 = request.form.get("q3")
        q4 = request.form.get("q4")
        q5 = request.form.get("q5")
        q6 = request.form.get("q6")
        q7 = request.form.get("q7")


        file = request.files["file"]
        if file:
            input_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(input_path)  # Save the original image


            # Process with quantum image processing
            processed_path = quantum_process_image(input_path)
            
            # Predict tumor type using the processed image
            label, confidence = predict_image(processed_path)


            # Adjust confidence based on questionnaire
            confidence1 = confidence
            if (label == 'Tumor'):
                if(int(q2) > 30):
                    confidence1 = confidence1 + 0.5
                if(int(q4) * int(q5)**2 * 703 > 25):  # Fixed the XOR (^) operator to proper exponentiation (**)
                    confidence1 = confidence1 + 0.5
                if(q6 == "yes"):
                    confidence1 = confidence1 + 0.5
                if(q7 == "yes"):
                    confidence1 = confidence1 + 0.5
            else:  # No Tumor case
                if(int(q2) > 30):
                    confidence1 = confidence1 - 0.5
                if(int(q4) * int(q5)**2 * 703 > 25):
                    confidence1 = confidence1 - 0.5
                if(q6 == "yes"):
                    confidence1 = confidence1 - 0.5
                if(q7 == "yes"):
                    confidence1 = confidence1 - 0.5


            # Redirect to results page with processed image
            return render_template("result.html", image_url=processed_path, label=label, confidence=confidence1)


    return render_template("main.html")




@app.route("/download/<filename>")
def download_file(filename):
    return send_file(os.path.join(app.config["UPLOAD_FOLDER"], filename), as_attachment=True)




@app.route('/home')
def home():
    return render_template('home.html')




if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=80)
