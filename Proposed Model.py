import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Set GPU configuration if needed
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Paths for dataset and labels
DATASET_PATH = r"E:\WLASL (World Level American Sign Language) Video\videos"
LABELS_FILE = r"E:\WLASL (World Level American Sign Language) Video\wlasl_class_list.txt"

# Parameters and hyperparameters
IMG_SIZE = 128
FRAMES = 16
BATCH_SIZE = 32
EPOCHS = 30
NUM_CLASSES = 100

# Function to extract frames from videos
def extract_frames(video_path, num_frames):
    frames = []
    cap = cv2.VideoCapture(video_path)  # Open video file
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // num_frames, 1)

    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()  # Read the frame
        if ret:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frames.append(frame)
        if len(frames) == num_frames:
            break

    cap.release()

    while len(frames) < num_frames:
        frames.append(np.zeros((IMG_SIZE, IMG_SIZE, 3)))

    return np.array(frames)

# Data loading and preprocessing
print("Loading and preprocessing data...")
data, labels = [], []
label_map = {}


with open(LABELS_FILE, "r") as f:
    for line in f:
        label, idx = line.strip().split()
        label_map[label] = int(idx)



for label, idx in label_map.items():
    label_path = os.path.join(DATASET_PATH, label)
    if not os.path.exists(label_path):
        continue

    # Load all videos under each label
    for video in os.listdir(label_path):
        video_path = os.path.join(label_path, video)
        frames = extract_frames(video_path, FRAMES)
        data.append(frames)
        labels.append(idx)

data = np.array(data)
labels = np.array(labels)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Normalize the pixel values to [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

# Data augmentation (Optional)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2)
])

# Hybrid CNN Model (DTCN + FSF3)
def create_hybrid_cnn():
    input_layer = layers.Input(shape=(FRAMES, IMG_SIZE, IMG_SIZE, 3))

    # DTCN Branch (Dynamic Temporal Convolution Network)
    dtcn = layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu'))(input_layer)
    dtcn = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(dtcn)
    dtcn = layers.TimeDistributed(layers.Flatten())(dtcn)

    # FSF3 Branch (FAST-SIFT Feature Fusion Framework)
    fsf3 = layers.TimeDistributed(layers.Conv2D(64, (5, 5), activation='relu'))(input_layer)
    fsf3 = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(fsf3)
    fsf3 = layers.TimeDistributed(layers.Flatten())(fsf3)

    # Fusion of both branches
    fusion = layers.Concatenate()([dtcn, fsf3])
    fusion = layers.LSTM(128, return_sequences=False)(fusion)

    return models.Model(inputs=input_layer, outputs=fusion)

# Transformer Layers (Symbolic and Progressive Transformers)
class SymbolicTransformer(layers.Layer):
    def __init__(self, num_heads, key_dim):
        super(SymbolicTransformer, self).__init__()
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)

    def call(self, inputs):
        return self.attention(inputs, inputs)

class ProgressiveTransformer(layers.Layer):
    def __init__(self, units):
        super(ProgressiveTransformer, self).__init__()
        self.dense1 = layers.Dense(units, activation='relu')
        self.dense2 = layers.Dense(units, activation='relu')

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

# Complete Model with Hybrid CNN + Transformer Layers
def create_complete_model():
    hybrid_cnn = create_hybrid_cnn()
    transformer_input = layers.Reshape((FRAMES, -1))(hybrid_cnn.output)

    symbolic = SymbolicTransformer(num_heads=4, key_dim=64)(transformer_input)

    progressive = ProgressiveTransformer(units=256)(symbolic)

    # Final output layer
    output = layers.Dense(NUM_CLASSES, activation='softmax')(progressive)

    return models.Model(inputs=hybrid_cnn.input, outputs=output)

# Compile and summarize the model
model = create_complete_model()
model.compile(optimizer=optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Training the model
history = model.fit(x_train, y_train, validation_split=0.2, epochs=EPOCHS, batch_size=BATCH_SIZE)

# Evaluate the model on the test set
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Print classification report for evaluation metrics
print("Classification Report:\n", classification_report(y_true, y_pred_classes))

# Plotting the accuracy and loss curves
plt.figure(figsize=(12, 6))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Save the trained model
model.save("hybrid_cnn_transformer_model.h5")
