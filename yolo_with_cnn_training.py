import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, Callback, LearningRateScheduler
from tensorflow.keras.applications import ResNet50
from PIL import Image, ImageFile
import datetime
import csv

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Function to validate and identify bad images and remove them
def clean_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                img = Image.open(file_path)
                img.verify()  # Verify that the file is an image
            except (IOError, SyntaxError) as e:
                print(f"Removing bad file: {file_path}")
                os.remove(file_path)  # Remove the bad file

# Clean up bad images in the train and validation directories
clean_directory('data/train')
clean_directory('data/validation')

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,  # Giảm từ 60
    width_shift_range=0.3,  # Giảm từ 0.4
    height_shift_range=0.3,  # Giảm từ 0.4
    shear_range=0.3,         # Giảm từ 0.4
    zoom_range=0.3,          # Giảm từ 0.4
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.7, 1.3],  # Điều chỉnh lại từ [0.5, 1.5]
    channel_shift_range=50.0    # Giảm từ 100.0
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(150, 150),
    batch_size=64,  # Increased batch size
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    'data/validation',
    target_size=(150, 150),
    batch_size=64,  # Increased batch size
    class_mode='categorical'
)

# Define the model
checkpoint_path = 'best_model_s1.keras'
initial_epoch = 0  # Initialize initial_epoch

if os.path.exists(checkpoint_path):
    print(f"Loading model from checkpoint: {checkpoint_path}")
    model = tf.keras.models.load_model(checkpoint_path)
else:
    print("No checkpoint found, training from scratch.")
    
    # Using a Pre-trained Model (ResNet50)
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    
    # Unfreeze the last layers for fine-tuning
    for layer in base_model.layers[-100:]:  # Unfreeze more layers
        layer.trainable = True

    # Add custom layers on top of the base model
    model = Sequential([
        base_model,
        Conv2D(256, (3, 3), activation='relu', padding='same'),  # Added Conv2D layer
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu', kernel_regularizer=l2(0.0001)),  # Reduced L2 regularization
        BatchNormalization(),
        Dropout(0.4),  # Tăng nhẹ dropout để giảm overfitting
        Dense(train_generator.num_classes, activation='softmax')
    ])

    # Compile model with a higher learning rate
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  # Reduced learning rate
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# Setting up TensorBoard logging
log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Custom callback to save best validation accuracy and other metrics
class MetricsCallback(Callback):
    def __init__(self, filename='metrics.csv', best_acc_file='best_val_accuracy_s1.txt'):
        super(MetricsCallback, self).__init__()
        self.filename = filename
        self.best_acc_file = best_acc_file
        self.best_val_accuracy = 0.0

        # Đọc giá trị val_accuracy tốt nhất từ tệp nếu nó tồn tại
        if os.path.exists(self.best_acc_file):
            with open(self.best_acc_file, 'r') as f:
                line = f.readline()
                self.best_val_accuracy = float(line.split()[-1])
                print(f"Loaded best validation accuracy: {self.best_val_accuracy}")

        # Mở tệp csv và ghi header
        self.file = open(self.filename, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(['Epoch', 'Train Loss', 'Train Accuracy', 'Val Loss', 'Val Accuracy'])

    def on_epoch_end(self, epoch, logs=None):
        val_accuracy = logs.get('val_accuracy')
        train_loss = logs.get('loss')
        train_accuracy = logs.get('accuracy')
        val_loss = logs.get('val_loss')

        # So sánh và cập nhật giá trị val_accuracy tốt nhất
        if val_accuracy and val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = val_accuracy
            with open(self.best_acc_file, 'w') as f:
                f.write(f'Best validation accuracy: {self.best_val_accuracy:.4f}\n')
            print(f"Updated best validation accuracy: {self.best_val_accuracy}")

        # Ghi các giá trị vào tệp CSV
        self.writer.writerow([epoch + 1, train_loss, train_accuracy, val_loss, val_accuracy])

    def on_train_end(self, logs=None):
        self.file.close()  # Đóng tệp khi huấn luyện kết thúc

# Early Stopping and Model Checkpoint
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)  # Tăng patience
model_checkpoint = ModelCheckpoint(
    checkpoint_path,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True
)

# Reduce learning rate when a metric has stopped improving
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=8, min_lr=1e-6)  # Tăng patience và giảm factor

# Learning Rate Scheduler
lr_scheduler = LearningRateScheduler(lambda epoch: 1e-3 * 10**(-epoch / 20))

# Custom callback for metrics
metrics_callback = MetricsCallback()

# Training the model with increased epochs and maxed-out parameters
model.fit(
    train_generator,
    epochs=50,  # Bắt đầu với 30 epoch trước, có thể tăng sau
    validation_data=validation_generator,
    callbacks=[early_stopping, model_checkpoint, tensorboard_callback, reduce_lr, lr_scheduler, metrics_callback],
    initial_epoch=initial_epoch
)

model = tf.keras.models.load_model('best_model_s1.keras')
model.save('disease_diagnosis_model_final_s1.keras')
