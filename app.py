from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
import os
import shutil
import subprocess

app = Flask(__name__)
CORS(app)

# Tải mô hình từ tệp .keras
model = load_model('D:/ai-demo/disease_diagnosis_model_final.keras')

def prepare_image(image_path):
    img = load_img(image_path, target_size=(150, 150))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

def train_model_if_new_images():
    upload_dir = 'uploads'
    if not os.path.exists(upload_dir) or len(os.listdir(upload_dir)) == 0:
        print("No new images to train.")
        return  # Nếu không có ảnh mới, không thực hiện huấn luyện

    print("New images found. Starting training...")
    
    # Chuyển các ảnh từ uploads vào thư mục train
    for file_name in os.listdir(upload_dir):
        file_path = os.path.join(upload_dir, file_name)
        
        # Dự đoán lớp của ảnh mới
        img = prepare_image(file_path)
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions, axis=1)[0]
        probability = np.max(predictions) * 100  # Xác suất của dự đoán cao nhất

        # Lấy tên lớp
        train_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(
            'data/train',
            target_size=(150, 150),
            batch_size=32,
            class_mode='categorical'
        )
        class_labels = list(train_generator.class_indices.keys())
        predicted_label = class_labels[predicted_class]

        # In thông tin dự đoán ra console
        print(f"Predicted: {predicted_label}, Probability: {probability:.2f}%")
        
        # Di chuyển ảnh vào thư mục train và validation
        train_dir = os.path.join('data/train', predicted_label)
        validate_dir = os.path.join('data/validation', predicted_label)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(validate_dir, exist_ok=True)
        
        shutil.move(file_path, os.path.join(train_dir, file_name))
        shutil.copy(os.path.join(train_dir, file_name), os.path.join(validate_dir, file_name))
        
        # Xóa ảnh sau khi đã di chuyển
        os.remove(os.path.join(train_dir, file_name))
    
    # Sau khi di chuyển ảnh, huấn luyện lại mô hình
    subprocess.run([os.path.join(os.environ['VIRTUAL_ENV'], 'Scripts', 'python.exe'), "cnn_finetune.py"])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        img = prepare_image(file_path)
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions, axis=1)[0]
        probability = np.max(predictions) * 100  # Xác suất của dự đoán cao nhất

        train_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(
            'data/train',
            target_size=(150, 150),
            batch_size=32,
            class_mode='categorical'
        )
        class_labels = list(train_generator.class_indices.keys())
        predicted_label = class_labels[predicted_class]

        # In thông tin dự đoán ra console
        print(f"Predicted: {predicted_label}, Probability: {probability:.2f}%")

        train_dir = os.path.join('data/train', predicted_label)
        validate_dir = os.path.join('data/validation', predicted_label)

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(validate_dir, exist_ok=True)

        shutil.copy(file_path, os.path.join(train_dir, file.filename))
        shutil.copy(file_path, os.path.join(validate_dir, file.filename))

        # Xóa ảnh sau khi đã di chuyển
        os.remove(file_path)

        return jsonify({'prediction': predicted_label}), 200

@app.route('/train', methods=['POST'])
def train():
    train_model_if_new_images()
    return jsonify({'message': 'Training completed if there were new images.'}), 200

if __name__ == '__main__':
    app.run(debug=True, port=8000)
