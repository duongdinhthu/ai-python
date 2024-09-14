import os
import cv2
import numpy as np
from ultralytics import YOLO
import albumentations as A
from sklearn.model_selection import KFold

# Khởi tạo mô hình YOLOv8 với mô hình pre-trained
yolo_model = YOLO('yolov8l.pt')  # Sử dụng YOLOv8l (Large)

# Augmentation mạnh mẽ
augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=20, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=20, p=0.5),
    A.Resize(640, 640)  # Đảm bảo kích thước đầu vào cho mô hình YOLO
])

# Hàm phát hiện đối tượng và chỉ giữ lại vùng da bị nhiễm bệnh
def detect_and_filter_skin(image_path, processed_dir):
    # Lấy tên lớp từ tên thư mục
    class_name = os.path.basename(os.path.dirname(image_path))

    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        print(f"Không thể đọc ảnh: {image_path}")
        return

    # Áp dụng Augmentation
    image_aug = augmentation(image=image)['image']

    # Dự đoán và lấy các bounding boxes
    results = yolo_model.predict(image_aug)
    boxes = results[0].boxes
    
    # Tạo một ảnh nền đen cùng kích thước với ảnh gốc
    mask = np.zeros_like(image_aug)
    
    # Kiểm tra nếu phát hiện được đối tượng
    if len(boxes) > 0:
        for box in boxes:
            # Lấy tọa độ x1, y1, x2, y2 từ box
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Đảm bảo tọa độ nằm trong giới hạn của ảnh
            if x1 >= 0 and y1 >= 0 and x2 <= image_aug.shape[1] and y2 <= image_aug.shape[0]:
                # Tạo mặt nạ chỉ giữ lại vùng da bị nhiễm bệnh
                mask[y1:y2, x1:x2] = image_aug[y1:y2, x1:x2]

                # Tạo thư mục lưu trữ ảnh đã phân tích nếu chưa có
                class_dir = os.path.join(processed_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)

                # Lưu ảnh đã xử lý vào thư mục
                save_path = os.path.join(class_dir, os.path.basename(image_path))
                cv2.imwrite(save_path, mask)
                print(f"Đã lưu ảnh vào {save_path}")

                # Xóa ảnh gốc sau khi đã lưu ảnh mới nếu tệp tồn tại
                if os.path.exists(image_path):
                    os.remove(image_path)
            else:
                print(f"Tọa độ của box không hợp lệ: {box.xyxy[0]}")
    else:
        print(f"Không phát hiện đối tượng trong {image_path}")
        if os.path.exists(image_path):
            os.remove(image_path)  # Xóa ảnh gốc nếu không phát hiện được đối tượng

# Đường dẫn đến thư mục train và validation
train_directory = 'data/train'
validation_directory = 'data/validation'
temp_directory = 'data/temp_processed'  # Tạm thời lưu ảnh đã xử lý ở đây

# Kiểm tra xem có ảnh mới không
new_images_detected = False
batch_size = 16
current_batch = 0

# Lặp qua tất cả các ảnh trong thư mục train
for root, dirs, files in os.walk(train_directory):
    for file in files:
        if file.endswith(('.jpg', '.jpeg', '.png')):
            image_full_path = os.path.join(root, file)
            detect_and_filter_skin(image_full_path, temp_directory)
            new_images_detected = True
            current_batch += 1

            if current_batch % batch_size == 0:
                # Lưu checkpoint sau mỗi batch
                checkpoint_path = f'checkpoints/model_batch_{current_batch}.pt'
                yolo_model.save(checkpoint_path)
                print(f'Checkpoint saved at batch {current_batch}: {checkpoint_path}')

if new_images_detected:
    print("Ảnh mới đã được xử lý và lưu vào thư mục train và validation.")
    
    # Cross-Validation Setup
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for train_idx, val_idx in kfold.split(os.listdir(train_directory)):
        yolo_model.train(data='data.yaml', epochs=10, imgsz=640, lr0=0.01, batch=16, resume=True)  # Thêm 'resume=True'
else:
    print("Không có ảnh mới, không cần huấn luyện lại mô hình.")

print("Quá trình phát hiện và lưu ảnh đã hoàn thành.")

# Model Ensemble (Nếu có thêm các mô hình khác)
yolo_model2 = YOLO('yolov8x.pt')  # Sử dụng một mô hình khác để kết hợp

# Lấy dự đoán từ cả hai mô hình
def ensemble_predict(image):
    results1 = yolo_model.predict(image)
    results2 = yolo_model2.predict(image)
    
    # Kết hợp kết quả
    final_results = (results1 + results2) / 2
    return final_results
