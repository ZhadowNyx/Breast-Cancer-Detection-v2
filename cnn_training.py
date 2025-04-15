from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, f1_score, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Define data paths
train_data_path = r'E:\1_Work_Files\6_Project - Breast Cancer Detection\BreaKHis_v1\flattened_split\train'
val_data_path = r'E:\1_Work_Files\6_Project - Breast Cancer Detection\BreaKHis_v1\flattened_split\val'
test_data_path = r'E:\1_Work_Files\6_Project - Breast Cancer Detection\BreaKHis_v1\flattened_split\test'

# ImageDataGenerators for data preprocessing and augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalize pixel values to [0, 1]
    rotation_range=20,  # Rotate images up to 20 degrees
    width_shift_range=0.2,  # Horizontally shift images by 20% of width
    height_shift_range=0.2,  # Vertically shift images by 20% of height
    zoom_range=0.2,  # Zoom in/out by 20%
    horizontal_flip=True,  # Randomly flip images horizontally
)

# Data generators for train, validation, and test datasets
train_data = datagen.flow_from_directory(
    train_data_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=True
)

val_data = datagen.flow_from_directory(
    val_data_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

test_data = datagen.flow_from_directory(
    test_data_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Build the CNN model
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Sigmoid for binary classification
])

# Compile the CNN model
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'AUC', 'Precision', 'Recall'])

# Early stopping and learning rate reduction callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=1e-6)

# Train the CNN model with validation and early stopping
history = cnn_model.fit(
    train_data,
    steps_per_epoch=train_data.samples // train_data.batch_size,
    epochs=50,
    validation_data=val_data,
    validation_steps=val_data.samples // val_data.batch_size,
    callbacks=[early_stopping, lr_reduction]
)

# Evaluate the model on the test set after training
test_labels = test_data.classes  # True labels for the test set
test_preds = cnn_model.predict(test_data, steps=test_data.samples // test_data.batch_size + 1).ravel()

# Convert probabilities to binary predictions
test_preds_binary = (test_preds > 0.5).astype(int)

# Classification report
print("\nClassification Report on Test Set:")
print(classification_report(test_labels, test_preds_binary, target_names=['Benign', 'Malignant']))

# AUC-ROC on test set
auc_roc = roc_auc_score(test_labels, test_preds)
print(f"\nAUC-ROC on Test Set: {auc_roc:.4f}")

# Confusion Matrix on Test Set
conf_matrix = confusion_matrix(test_labels, test_preds_binary)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.title('Confusion Matrix - Test Set')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Save the trained model
cnn_model.save('cnn_model_breast_cancer.h5')
print("\nModel saved as 'cnn_model_breast_cancer.h5'")

# Visualize training history
def plot_training_history(history):
    # Plot accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Plot loss
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

plot_training_history(history)

# F1 Score calculation
f1 = f1_score(test_labels, test_preds_binary)
print(f"\nF1 Score on Test Set: {f1:.4f}")

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(test_labels, test_preds)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()
