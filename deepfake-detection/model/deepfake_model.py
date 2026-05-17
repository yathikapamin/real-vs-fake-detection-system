"""
DeepFake Detection Model Architecture
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2


class DeepFakeDetector:
    """
    Deep Learning model for detecting deepfake images and videos
    Supports both 128x128 and 224x224 input sizes
    """
    
    def __init__(self, input_shape=(224, 224, 3), use_pretrained=True):
        """
        Initialize the DeepFake Detector model
        
        Args:
            input_shape: Shape of input images (height, width, channels)
                        Recommended: (128, 128, 3) or (224, 224, 3)
            use_pretrained: Whether to use transfer learning with pre-trained model
        """
        self.input_shape = input_shape
        self.use_pretrained = use_pretrained
        self.model = None
        
    def build_custom_cnn(self):
        """
        Build a custom CNN architecture for deepfake detection
        Automatically adjusts layers based on input size
        """
        # Determine number of conv blocks based on input size
        # For 128x128: use 5 blocks, for 224x224: use 4 blocks
        height = self.input_shape[0]
        use_five_blocks = (height <= 128)
        
        layers_list = [
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                         input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
        ]
        
        # Add fifth block for smaller input sizes (128x128)
        if use_five_blocks:
            layers_list.extend([
                layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.3),
            ])
        
        # Add dense layers
        layers_list.extend([
            # Flatten and Dense Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output Layer - Single neuron with sigmoid for binary classification
            layers.Dense(1, activation='sigmoid')
        ])
        
        model = models.Sequential(layers_list)
        
        return model
    
    def build_transfer_learning_model(self, base_model='efficientnet'):
        """
        Build a model using transfer learning
        
        Args:
            base_model: Choice of pre-trained model ('efficientnet' or 'mobilenet')
        """
        # Load pre-trained base model
        if base_model == 'efficientnet':
            base = EfficientNetB0(
                include_top=False,
                weights='imagenet',
                input_shape=self.input_shape
            )
        else:
            base = MobileNetV2(
                include_top=False,
                weights='imagenet',
                input_shape=self.input_shape
            )
        
        # Freeze base model layers
        base.trainable = False
        
        # Build the complete model
        model = models.Sequential([
            base,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        
        return model
    
    def build_model(self, base_model='efficientnet'):
        """
        Build the complete model based on configuration
        
        Args:
            base_model: Choice of pre-trained model for transfer learning
        """
        if self.use_pretrained:
            self.model = self.build_transfer_learning_model(base_model)
        else:
            self.model = self.build_custom_cnn()
        
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile the model with optimizer and loss function
        
        Args:
            learning_rate: Learning rate for the optimizer
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        
        return self.model
    
    def get_model_summary(self):
        """
        Print model summary
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        
        return self.model.summary()
    
    def save_model(self, filepath='deepfake_detector.h5'):
        """
        Save the trained model
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not built yet.")
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='deepfake_detector.h5'):
        """
        Load a trained model
        
        Args:
            filepath: Path to the saved model
        """
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        return self.model


if __name__ == "__main__":
    # Example usage
    detector = DeepFakeDetector(use_pretrained=True)
    model = detector.build_model(base_model='efficientnet')
    model = detector.compile_model(learning_rate=0.001)
    
    print("Model Architecture:")
    detector.get_model_summary()
