import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
from datetime import datetime
import json

class MultiClass3ClassModel:
    """
    3-Class Brain Tumor Classification Model
    Classes: Glioma, Meningioma, Pituitary (NO No Tumor class)
    
    Based on proven 4-class architecture adapted for tumor-only classification
    """
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        self.class_names = ['Glioma', 'Meningioma', 'Pituitary']  # Only tumor classes
        
        # Medical-specific parameters
        self.medical_config = {
            'batch_size': 32,
            'epochs': 50,
            'patience': 15,
            'learning_rate': 0.001,
            'validation_split': 0.2,
            'test_split': 0.1
        }
    
    def create_model(self):
        """
        Create 3-class multiclass model based on proven 4-class architecture
        Optimized for tumor-only classification (no healthy brain confusion)
        """
        print("🏗️ Creating 3-Class Multiclass Model...")
        print("📊 Classes: Glioma, Meningioma, Pituitary (Tumor Types Only)")
        
        model = models.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # Block 1: Initial Feature Extraction
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_1'),
            layers.BatchNormalization(name='bn1_1'),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_2'),
            layers.BatchNormalization(name='bn1_2'),
            layers.MaxPooling2D(pool_size=(2, 2), name='pool1'),
            layers.Dropout(0.25, name='dropout1'),
            
            # Block 2: Mid-Level Features
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_1'),
            layers.BatchNormalization(name='bn2_1'),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_2'),
            layers.BatchNormalization(name='bn2_2'),
            layers.MaxPooling2D(pool_size=(2, 2), name='pool2'),
            layers.Dropout(0.25, name='dropout2'),
            
            # Block 3: High-Level Features
            layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_1'),
            layers.BatchNormalization(name='bn3_1'),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_2'),
            layers.BatchNormalization(name='bn3_2'),
            layers.MaxPooling2D(pool_size=(2, 2), name='pool3'),
            layers.Dropout(0.3, name='dropout3'),
            
            # Block 4: Abstract Features
            layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4_1'),
            layers.BatchNormalization(name='bn4_1'),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4_2'),
            layers.BatchNormalization(name='bn4_2'),
            layers.MaxPooling2D(pool_size=(2, 2), name='pool4'),
            layers.Dropout(0.3, name='dropout4'),
            
            # Classification Head - Modified for 3-class tumor discrimination
            layers.GlobalAveragePooling2D(name='global_avg_pool'),
            layers.Dense(256, activation='relu', name='dense1'),
            layers.BatchNormalization(name='bn_dense1'),
            layers.Dropout(0.5, name='dropout_dense1'),
            
            layers.Dense(128, activation='relu', name='dense2'),
            layers.BatchNormalization(name='bn_dense2'),
            layers.Dropout(0.4, name='dropout_dense2'),
            
            # 3-class tumor output with softmax
            layers.Dense(self.num_classes, activation='softmax', name='tumor_classification')
        ])
        
        self.model = model
        
        # Print model summary
        print("\n📋 Model Architecture Summary:")
        self.model.summary()
        
        # Calculate parameters
        total_params = self.model.count_params()
        print(f"\n📊 Total Parameters: {total_params:,}")
        print(f"💾 Model Size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
        
        return model
    
    def compile_model(self):
        """
        Compile model with appropriate loss and metrics for 3-class tumor classification
        """
        print("\n⚙️ Compiling Model for 3-Class Tumor Classification...")
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.medical_config['learning_rate']),
            loss='categorical_crossentropy',  # For multiclass classification
            metrics=[
                'accuracy',
                keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )
        
        print("✅ Model compiled successfully!")
        print("🎯 Loss: Categorical Crossentropy")
        print("📊 Metrics: Accuracy, Top-2 Accuracy, Precision, Recall")
    
    def setup_data_augmentation(self):
        """
        Setup medical-appropriate data augmentation for tumor classification
        """
        print("\n🔄 Setting up Medical Data Augmentation...")
        
        # Training data augmentation - conservative for medical imaging
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,           # Conservative rotation
            width_shift_range=0.1,       # Slight shifts
            height_shift_range=0.1,
            horizontal_flip=True,        # Brain symmetry allows flipping
            zoom_range=0.1,              # Slight zoom
            brightness_range=[0.9, 1.1], # Slight brightness variation
            fill_mode='nearest'
        )
        
        # Validation/Test data - only rescaling
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        print("✅ Data augmentation configured:")
        print("  - Rotation: ±15°")
        print("  - Shifts: ±10%")
        print("  - Zoom: ±10%")
        print("  - Brightness: ±10%")
        print("  - Horizontal flip enabled")
        
        return train_datagen, val_test_datagen
    
    def prepare_callbacks(self, model_name="multiclass_3class", output_dir="."):
        """
        Setup training callbacks for optimal training
        """
        print("\n📞 Setting up Training Callbacks...")
        
        callbacks_list = [
            # Early stopping
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=self.medical_config['patience'],
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Model checkpoint
            callbacks.ModelCheckpoint(
                filepath=os.path.join(output_dir, f'best_{model_name}_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            
            # CSV logger
            callbacks.CSVLogger(
                os.path.join(output_dir, f'{model_name}_training_log.csv'),
                append=False
            )
        ]
        
        print("✅ Callbacks configured:")
        print(f"  - Early stopping: patience={self.medical_config['patience']}")
        print("  - Learning rate reduction: factor=0.5, patience=8")
        print("  - Model checkpointing enabled")
        print("  - Training logging enabled")
        
        return callbacks_list
    
    def load_and_preprocess_data(self, data_path):
        """
        Load and preprocess 3-class tumor dataset (NO No Tumor class)
        Expected structure:
        data_path/
        ├── glioma/
        ├── meningioma/
        └── pituitary/
        """
        print(f"\n📁 Loading 3-Class Tumor Dataset from: {data_path}")
        
        # Check if path exists
        if not os.path.exists(data_path):
            raise ValueError(f"Data path does not exist: {data_path}")
        
        images = []
        labels = []
        class_counts = {}
        
        # Define class mapping - ONLY tumor classes
        class_to_idx = {
            'glioma': 0,
            'meningioma': 1,  
            'pituitary': 2
        }
        
        for class_name, class_idx in class_to_idx.items():
            class_path = os.path.join(data_path, class_name)
            if not os.path.exists(class_path):
                print(f"⚠️ Warning: {class_name} folder not found at {class_path}")
                continue
            
            class_images = []
            image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            print(f"📊 Loading {class_name}: {len(image_files)} images")
            
            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                try:
                    # Load and preprocess image
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (224, 224))
                    
                    images.append(img)
                    labels.append(class_idx)
                    class_images.append(img)
                    
                except Exception as e:
                    print(f"⚠️ Error loading {img_path}: {e}")
                    continue
            
            class_counts[class_name] = len(class_images)
        
        # Convert to numpy arrays
        X = np.array(images, dtype=np.float32)
        y = np.array(labels)
        
        print(f"\n📊 Tumor Dataset Statistics:")
        print(f"  Total tumor images: {len(X):,}")
        for class_name, count in class_counts.items():
            percentage = (count / len(X)) * 100
            print(f"  {class_name.capitalize()}: {count:,} images ({percentage:.1f}%)")
        
        # Convert labels to categorical
        y_categorical = utils.to_categorical(y, num_classes=self.num_classes)
        
        return X, y_categorical, class_counts
    
    def create_data_splits(self, X, y, test_size=0.1, val_size=0.2):
        """
        Create train/validation/test splits with stratification
        """
        print(f"\n✂️ Creating Data Splits:")
        print(f"  Test: {test_size*100:.0f}%")
        print(f"  Validation: {val_size*100:.0f}%") 
        print(f"  Training: {(1-test_size-val_size)*100:.0f}%")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y.argmax(axis=1), random_state=42
        )
        
        # Second split: training and validation
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            stratify=y_temp.argmax(axis=1), random_state=42
        )
        
        print(f"\n📊 Final Split Sizes:")
        print(f"  Training: {len(X_train):,} images")
        print(f"  Validation: {len(X_val):,} images")
        print(f"  Test: {len(X_test):,} images")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_model(self, X_train, y_train, X_val, y_val, output_dir="."):
        """
        Train the 3-class tumor multiclass model
        """
        print("\n🚀 Starting 3-Class Tumor Classification Training...")
        print(f"📊 Training samples: {len(X_train):,}")
        print(f"📊 Validation samples: {len(X_val):,}")
        
        # Setup data augmentation
        train_datagen, val_datagen = self.setup_data_augmentation()
        
        # Prepare callbacks
        callbacks_list = self.prepare_callbacks("multiclass_3class", output_dir)
        
        # Calculate class weights for balanced training
        y_train_labels = y_train.argmax(axis=1)
        class_weights = {}
        for i in range(self.num_classes):
            class_count = np.sum(y_train_labels == i)
            if class_count > 0:
                class_weights[i] = len(y_train_labels) / (self.num_classes * class_count)
            else:
                # Handle missing classes by setting weight to 0
                class_weights[i] = 0.0
        
        print(f"\n⚖️ Class Weights for Balanced Training:")
        for i, weight in class_weights.items():
            print(f"  {self.class_names[i]}: {weight:.3f}")
        
        # Training generators
        train_generator = train_datagen.flow(
            X_train, y_train,
            batch_size=self.medical_config['batch_size'],
            shuffle=True
        )
        
        val_generator = val_datagen.flow(
            X_val, y_val,
            batch_size=self.medical_config['batch_size'],
            shuffle=False
        )
        
        # Start training
        start_time = datetime.now()
        print(f"⏰ Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.history = self.model.fit(
            train_generator,
            epochs=self.medical_config['epochs'],
            validation_data=val_generator,
            callbacks=callbacks_list,
            class_weight=class_weights,
            verbose=1
        )
        
        end_time = datetime.now()
        training_time = end_time - start_time
        print(f"\n✅ Training completed!")
        print(f"⏱️ Total training time: {training_time}")
        
        return self.history
    
    def evaluate_model(self, X_test, y_test, output_dir="."):
        """
        Comprehensive evaluation of the trained model
        """
        print("\n📊 Evaluating 3-Class Tumor Classification Model...")
        
        # Load best model
        best_model_path = os.path.join(output_dir, 'best_multiclass_3class_model.h5')
        if os.path.exists(best_model_path):
            print("📁 Loading best saved model...")
            self.model = keras.models.load_model(best_model_path)
        
        # Predictions
        y_pred_proba = self.model.predict(X_test / 255.0, verbose=1)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Overall accuracy
        test_loss, test_accuracy, test_top2, test_precision, test_recall = self.model.evaluate(
            X_test / 255.0, y_test, verbose=0
        )
        
        print(f"\n🎯 Test Results:")
        print(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Top-2 Accuracy: {test_top2:.4f} ({test_top2*100:.2f}%)")
        print(f"  Precision: {test_precision:.4f}")
        print(f"  Recall: {test_recall:.4f}")
        
        # Classification report
        print(f"\n📋 Detailed Classification Report:")
        class_report = classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            digits=4
        )
        print(class_report)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'top_2_accuracy': test_top2,
            'precision': test_precision,
            'recall': test_recall,
            'predictions': y_pred,
            'true_labels': y_true,
            'prediction_probabilities': y_pred_proba,
            'confusion_matrix': cm,
            'classification_report': class_report
        }
    
    def plot_training_history(self, output_dir="."):
        """
        Plot training curves and save visualizations
        """
        if self.history is None:
            print("⚠️ No training history available")
            return
        
        print("\n📈 Creating Training Visualizations...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy - 3-Class Tumor Classification', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss
        ax2.plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        ax2.plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.set_title('Model Loss - 3-Class Tumor Classification', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Top-2 Accuracy
        ax3.plot(self.history.history['top_2_accuracy'], label='Training Top-2', linewidth=2)
        ax3.plot(self.history.history['val_top_2_accuracy'], label='Validation Top-2', linewidth=2)
        ax3.set_title('Top-2 Accuracy - 3-Class Tumor Classification', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Top-2 Accuracy')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Learning Rate (if available)
        if 'lr' in self.history.history:
            ax4.plot(self.history.history['lr'], linewidth=2, color='red')
            ax4.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Learning Rate')
            ax4.set_yscale('log')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Learning Rate\nHistory\nNot Available', 
                    transform=ax4.transAxes, ha='center', va='center', fontsize=12)
            ax4.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, 'multiclass_3class_training_history.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Training history saved: {save_path}")
        
        return fig
    
    def plot_confusion_matrix(self, confusion_matrix, output_dir="."):
        """
        Create and save confusion matrix visualization
        """
        print("\n🔍 Creating Confusion Matrix...")
        
        plt.figure(figsize=(10, 8))
        
        # Calculate percentages
        cm_percentage = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis] * 100
        
        # Create heatmap
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   cbar_kws={'label': 'Number of Samples'})
        
        plt.title('Confusion Matrix - 3-Class Tumor Classification', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        
        # Add accuracy text
        accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
        plt.figtext(0.5, 0.02, f'Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)', 
                   ha='center', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, 'multiclass_3class_confusion_matrix.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Confusion matrix saved: {save_path}")
        
        # Also create percentage version
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_percentage, annot=True, fmt='.1f', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   cbar_kws={'label': 'Percentage (%)'})
        
        plt.title('Confusion Matrix (Percentage) - 3-Class Tumor Classification', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        save_path_pct = os.path.join(output_dir, 'multiclass_3class_confusion_matrix_percentage.png')
        plt.savefig(save_path_pct, dpi=300, bbox_inches='tight')
        print(f"💾 Percentage confusion matrix saved: {save_path_pct}")


# Example usage function
def main():
    """
    Main function to demonstrate 3-class tumor classification training
    """
    print("🧠 3-Class Tumor Classification Training")
    print("=" * 60)
    
    # Initialize model
    model = MultiClass3ClassModel()
    
    # Create architecture
    model.create_model()
    
    # Compile model
    model.compile_model()


if __name__ == "__main__":
    main()