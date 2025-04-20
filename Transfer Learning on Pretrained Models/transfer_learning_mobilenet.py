from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Data paths
train_data_path = r'E:\1_Work_Files\6_Project - Breast Cancer Detection\BreaKHis_v1\flattened_split\train'
val_data_path = r'E:\1_Work_Files\6_Project - Breast Cancer Detection\BreaKHis_v1\flattened_split\val'
test_data_path = r'E:\1_Work_Files\6_Project - Breast Cancer Detection\BreaKHis_v1\flattened_split\test'

# Data preprocessing and augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

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

# Load MobileNet base model
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Add custom classification layers
transfer_model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
transfer_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# Train with frozen base
history_transfer = transfer_model.fit(
    train_data,
    steps_per_epoch=train_data.samples // train_data.batch_size,
    epochs=100,
    validation_data=val_data,
    validation_steps=val_data.samples // val_data.batch_size,
    callbacks=[early_stopping]
)

# Unfreeze and fine-tune
base_model.trainable = True
transfer_model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

history_finetune = transfer_model.fit(
    train_data,
    steps_per_epoch=train_data.samples // train_data.batch_size,
    epochs=100,
    validation_data=val_data,
    validation_steps=val_data.samples // val_data.batch_size,
    callbacks=[early_stopping]
)

# Evaluate on test set
test_labels = test_data.classes
test_preds = transfer_model.predict(test_data, steps=test_data.samples // test_data.batch_size + 1).ravel()
test_preds_binary = (test_preds > 0.5).astype(int)

# Classification report
print("\nClassification Report on Test Set:")
print(classification_report(test_labels, test_preds_binary, target_names=['Benign', 'Malignant']))

# Metrics
auc_roc = roc_auc_score(test_labels, test_preds)
print(f"\nAUC-ROC on Test Set: {auc_roc:.4f}")
print(f"Precision: {precision_score(test_labels, test_preds_binary):.4f}")
print(f"Recall: {recall_score(test_labels, test_preds_binary):.4f}")
print(f"F1 Score: {f1_score(test_labels, test_preds_binary):.4f}")

# Confusion matrix
conf_matrix = confusion_matrix(test_labels, test_preds_binary)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.title('Confusion Matrix - Test Set')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# ROC curve
fpr, tpr, _ = roc_curve(test_labels, test_preds)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# Save model
transfer_model.save('transfer_model_mobilenet_breast_cancer.h5')
print("\nFine-tuned transfer learning model saved as 'transfer_model_mobilenet_breast_cancer.h5'")

# Plot training history
def plot_training_history(history, history_finetune):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy (Frozen)')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy (Frozen)')
    plt.plot(history_finetune.history['accuracy'], label='Train Accuracy (Fine-tune)')
    plt.plot(history_finetune.history['val_accuracy'], label='Val Accuracy (Fine-tune)')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Train Loss (Frozen)')
    plt.plot(history.history['val_loss'], label='Val Loss (Frozen)')
    plt.plot(history_finetune.history['loss'], label='Train Loss (Fine-tune)')
    plt.plot(history_finetune.history['val_loss'], label='Val Loss (Fine-tune)')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

plot_training_history(history_transfer, history_finetune)
