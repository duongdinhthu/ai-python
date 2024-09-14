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

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Tải mô hình dự đoán từ tệp .keras đã được huấn luyện
model_path = 'D:/ai-demo/disease_diagnosis_model_final.keras'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Mô hình không tồn tại tại {model_path}")

model = load_model(model_path)

# Tải mô hình YOLO mà bạn đã sử dụng trước đây
yolo_model = YOLO('yolov8l.pt')  # Sử dụng YOLOv8l (Large)

# Tải file JSON chứa tư vấn và đơn thuốc
with open('D:/ai-demo/advice_and_prescriptions.json', 'r', encoding='utf-8') as f:
    advice_and_prescriptions = json.load(f)

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
