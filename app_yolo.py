import threading  # Dùng để khởi chạy chế độ nền
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import cv2
from ultralytics import YOLO
from PIL import Image
import albumentations as A
import json
from collections import Counter
from datetime import datetime
import requests

# Thông tin từ Azure Blob Storage (thay thế bằng SAS URL của bạn)
keras_model_url = "https://assetprojectai.blob.core.windows.net/private/disease_diagnosis_model_final.keras?sp=r&st=2024-09-14T08:24:23Z&se=2025-01-11T16:24:23Z&spr=https&sv=2022-11-02&sr=b&sig=Fn3WKJrQvpLMpjQfWjO7v%2FNn6kqTIEc%2B0tbPoAzslEM%3D"
yolo_model_url = "https://assetprojectai.blob.core.windows.net/private/yolov8l.pt?sp=r&st=2024-09-14T08:35:33Z&se=2025-01-15T16:35:33Z&spr=https&sv=2022-11-02&sr=b&sig=DYAJDTcz6fV5%2BU8JtFlJGQ0yJaEDuzPaXkKHZ%2BkhmhU%3D"

# Đường dẫn local để lưu tệp tải về
local_keras_model_path = 'disease_diagnosis_model_final.keras'
local_yolo_model_path = 'yolov8l.pt'

# Hàm để tải tệp từ URL
def download_file_from_url(url, local_path):
    response = requests.get(url)
    with open(local_path, 'wb') as file:
        file.write(response.content)

# Tải mô hình Keras từ Azure Blob Storage
if not os.path.exists(local_keras_model_path):
    download_file_from_url(keras_model_url, local_keras_model_path)

# Tải mô hình YOLO từ Azure Blob Storage
if not os.path.exists(local_yolo_model_path):
    download_file_from_url(yolo_model_url, local_yolo_model_path)

# Tải mô hình dự đoán từ tệp .keras đã được huấn luyện
model = load_model(local_keras_model_path)

# Tải mô hình YOLO mà bạn đã sử dụng trước đây
yolo_model = YOLO(local_yolo_model_path)  # Sử dụng YOLOv8l (Large)

# Đường dẫn đến tệp JSON trên GitHub
json_url = "https://raw.githubusercontent.com/duongdinhthu/ai-python/main/advice_and_prescriptions.json"

# Tải file JSON chứa tư vấn và đơn thuốc từ GitHub
response = requests.get(json_url)
advice_and_prescriptions = response.json()

# Khởi tạo ứng dụng Flask
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Augmentation mạnh mẽ
augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=20, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=20, p=0.5),
    A.Resize(640, 640)  # Đảm bảo kích thước đầu vào cho mô hình YOLO
])

def prepare_image(image, target_size=(150, 150)):
    if image.size != target_size:
        image = image.resize(target_size)
    img = img_to_array(image)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

def detect_and_filter_skin(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Không thể đọc ảnh: {image_path}")
        return []

    image_aug = augmentation(image=image)['image']
    results = yolo_model.predict(image_aug)
    boxes = results[0].boxes

    crops = []
    if len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if x1 >= 0 and y1 >= 0 and x2 <= image_aug.shape[1] and y2 <= image_aug.shape[0]:
                crop = image_aug[y1:y2, x1:x2]
                crop_image = Image.fromarray(crop)
                crops.append(crop_image)
            else:
                print(f"Tọa độ của box không hợp lệ: {box.xyxy[0]}")
    else:
        print(f"Không phát hiện đối tượng trong {image_path}")
    
    return crops

def calculate_severity(prediction_probabilities):
    """
    Hàm để tính toán mức độ nghiêm trọng.
    Chúng ta giả định rằng đầu vào là xác suất dự đoán từ mô hình.
    """
    try:
        severity_score = np.max(prediction_probabilities)
        print(f"Calculated severity score: {severity_score}")  # In ra log để kiểm tra giá trị
        return severity_score  # Trả về giá trị số thay vì chuỗi
    except Exception as e:
        print(f"Error calculating severity: {e}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'files' not in request.files:
            print("Error: 'files' not found in request")
            return jsonify({'error': 'No files part'}), 400

        files = request.files.getlist('files')
        print(f"Received {len(files)} files")

        if len(files) == 0:
            print("Error: No selected files")
            return jsonify({'error': 'No selected files'}), 400

        if not os.path.exists('uploads'):
            os.makedirs('uploads')

        all_predictions = []  # List to store predictions for all files
        saved_files = []  # Lưu đường dẫn của tất cả các file đã tải lên
        severity_scores = []  # List to store severity scores for each prediction

        for file in files:
            try:
                file_path = os.path.join('uploads', file.filename)
                file.save(file_path)
                saved_files.append(file_path)
                print(f"File saved to {file_path}")

                crops = detect_and_filter_skin(file_path)

                if crops:
                    for crop in crops:
                        img = prepare_image(crop)
                        prediction = model.predict(img)
                        predicted_class = np.argmax(prediction, axis=1)[0]
                        severity = calculate_severity(prediction)

                        if severity is not None:
                            severity_scores.append(float(severity))

                        train_datagen = ImageDataGenerator(rescale=1./255)
                        train_generator = train_datagen.flow_from_directory(
                            'D:/ai-demo/data/train',  
                            target_size=(150, 150),
                            batch_size=32,
                            class_mode='categorical'
                        )

                        class_labels = list(train_generator.class_indices.keys())
                        predicted_label = class_labels[predicted_class]

                        all_predictions.append(predicted_label)
                        print(f"Image: {file.filename}, Prediction: {predicted_label}, Severity: {severity}")
                else:
                    img = load_img(file_path, target_size=(150, 150))
                    img = prepare_image(img)
                    prediction = model.predict(img)
                    predicted_class = np.argmax(prediction, axis=1)[0]
                    severity = calculate_severity(prediction)

                    if severity is not None:
                        severity_scores.append(float(severity))

                    train_datagen = ImageDataGenerator(rescale=1./255)
                    train_generator = train_datagen.flow_from_directory(
                        'D:/ai-demo/data/train',
                        target_size=(150, 150),
                        batch_size=32,
                        class_mode='categorical'
                    )
                    class_labels = list(train_generator.class_indices.keys())
                    predicted_label = class_labels[predicted_class]

                    all_predictions.append(predicted_label)
                    print(f"Image: {file.filename}, Prediction: {predicted_label}, Severity: {severity}")
            except Exception as e:
                print(f"Error processing file {file.filename}: {e}")
        
        most_common_prediction = Counter(all_predictions).most_common(1)[0][0]  
        if severity_scores:
            most_common_severity = np.mean(severity_scores)
        else:
            most_common_severity = None

        advice = advice_and_prescriptions.get(most_common_prediction, {}).get('advice', 'No advice available.')
        prescription = advice_and_prescriptions.get(most_common_prediction, {}).get('prescription', 'No prescription available.')

        response = jsonify({
            'conclusion': most_common_prediction,
            'severity': most_common_severity, 
            'advice': advice,
            'prescription': prescription
        })

        return response, 200
    except Exception as e:
        print(f"Error in /predict route: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
