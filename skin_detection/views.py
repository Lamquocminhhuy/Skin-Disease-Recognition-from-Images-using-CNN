from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Load model
MODEL_PATH = 'models/HAM10000_model.h5'
model = load_model(MODEL_PATH)

# Class labels
CLASS_NAMES = ['Actinic Keratoses (akiec)', 'Basal Cell Carcinoma (bcc)', 'Benign Keratosis-like (bkl)', 
               'Dermatofibroma (df)', 'Melanoma (mel)', 'Melanocytic Nevi (nv)', 'Vascular Lesions (vasc)']

def upload_and_predict(request):
    context = {}
    
    if request.method == 'POST' and request.FILES.get('image', None):
        # Handle file upload
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage()
        file_path = fs.save(uploaded_file.name, uploaded_file)
        file_url = fs.url(file_path)

        # Preprocess the image
        img_path = fs.path(file_path)
        img = load_img(img_path, target_size=(128, 128))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        confidence = np.max(predictions)

        # Add results to context
        context['predicted_class'] = predicted_class
        context['confidence'] = f"{confidence * 100:.2f}%"
        context['image_url'] = file_url

    return render(request, 'upload.html', context)
