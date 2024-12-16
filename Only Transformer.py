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
    cap = cv2.VideoCapture(video_path)
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
    # Pad with zeros if there are fewer frames than required
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

# Iterate through each label and load associated videos
for label, idx in label_map.items():
    label_path = os.path.join(DATASET_PATH, label)
    if not os.path.exists(label_path):
        continue

    # Load all videos under each label
    for video in os.listdir(label_path):
        video_path = os.path.join(label_path, video)
        frames = extract_frames(video_path, FRAMES)  # Extract frames from video
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

# Transformer-based Model
def create_transformer_model():
    input_layer = layers.Input(shape=(FRAMES, IMG_SIZE, IMG_SIZE, 3))
    x = layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu'))(input_layer)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
    x = layers.TimeDistributed(layers.Flatten())(x)

    x = SymbolicTransformer(2, 32)(x)  # Add symbolic transformer
    x = ProgressiveTransformer(128)(x)  # Add progressive transformer

    x = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    return models.Model(inputs=input_layer, outputs=x)

# Compile and summarize the model
model = create_transformer_model()
model.compile(optimizer=optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Training the model
history = model.fit(x_train, y_train, validation_split=0.2, epochs=EPOCHS, batch_size=BATCH_SIZE)

# Evaluate the model on the test set
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)


print("Classification Report:\n", classification_report(y_true, y_pred_classes))


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
model.save("transformer_model.h5")
