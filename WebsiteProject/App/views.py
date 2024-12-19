import os
import numpy as np
import pandas as pd
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.conf import settings
from .models import PredictionResult
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load model and CSV data
MODEL_PATH = os.path.join(settings.BASE_DIR, 'App', 'models', 'web.keras')  # Adjusted to use absolute path
CSV_PATH = os.path.join(settings.BASE_DIR, 'App', 'data', 'final.csv')      # Adjusted to use absolute path

model = load_model(MODEL_PATH)
data = pd.read_csv(CSV_PATH)
class_names = sorted(data['label'].unique())

# Image preprocessing function
def preprocess_image(image_path, target_size=(224, 224)):
    try:
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img) / 255.0  # Normalize
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def upload_image(request):
    # Fetch all prediction results from the database
    prediction_results = PredictionResult.objects.all()

    if request.method == 'POST' and 'image' in request.FILES:
        # Handle the image upload and prediction logic
        uploaded_file = request.FILES['image']
        file_name = default_storage.save(uploaded_file.name, uploaded_file)
        file_path = os.path.join(settings.MEDIA_ROOT, file_name)

        input_image = preprocess_image(file_path)
        if input_image is None:
            return render(request, 'upload.html', {'error': 'Invalid image file. Please try again.'})

        # Make predictions
        predictions = model.predict(input_image)
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        predicted_label = class_names[predicted_class_idx]
        confidence = round(np.max(predictions) * 100, 2)
        benefits = data[data['label'] == predicted_label]['definition'].iloc[0]

        # Save the prediction result to the database
        PredictionResult.objects.create(
            label=predicted_label,
            confidence=confidence,
            benefits=benefits
        )

        # Pass the results to the result template
        context = {
            'predicted_label': predicted_label,
            'confidence': confidence,
            'benefits': benefits,
        }
        return render(request, 'result.html', context)

    # If no image is uploaded, just pass all results from the database
    return render(request, 'upload.html', {'prediction_results': prediction_results})

# Home view
def home(request):
    return render(request, 'home.html')
