import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
import tensorflow as tf

# Load dataset using pandas
csv_path = 'combined_dataset.csv'
dataset_folder = 'Dataset'
df = pd.read_csv(csv_path)

# Display the first few rows to check the data
print(df.head())

# Image preprocessing function
def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess an image for model input."""
    img = load_img(image_path, target_size=target_size)  # Load image with the target size
    img = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
    return img

# Prepare pairs and labels
X1, X2, Y = [], [], []

# Iterate through the rows of the dataframe to process image pairs
for _, row in df.iterrows():
    img1_path = os.path.join(dataset_folder, row['Image 1'])
    img2_path = os.path.join(dataset_folder, row['Image 2'])
    
    # Check if the images exist before processing
    if os.path.exists(img1_path) and os.path.exists(img2_path):
        X1.append(preprocess_image(img1_path))
        X2.append(preprocess_image(img2_path))
        
        # Determine the label based on the 'Winner' column
        # If image1 wins (Winner = 1), label is 0; otherwise, it's 1
        Y.append(0 if row['Winner'] == 1 else 1)

# Convert to numpy arrays (assuming you're using TensorFlow or PyTorch)
X1 = np.array(X1)
X2 = np.array(X2)
Y = np.array(Y)

# Check if the data is prepared correctly
print(f"Shape of X1: {X1.shape}")
print(f"Shape of X2: {X2.shape}")
print(f"Shape of Y: {Y.shape}")

# Split data into training and testing sets
X1_train, X1_test, X2_train, X2_test, Y_train, Y_test = train_test_split(X1, X2, Y, test_size=0.2, random_state=42)

# Create a simple CNN model for food image comparison
def create_model(input_shape=(224, 224, 3)):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Build the model
model = create_model()

# Print model summary to check the architecture
model.summary()

# Train the model
history = model.fit([X1_train, X2_train], Y_train, epochs=100, batch_size=64, validation_data=([X1_test, X2_test], Y_test))

# Evaluate the model on test data
test_loss, test_acc = model.evaluate([X1_test, X2_test], Y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Save the model
model.save('food_comparator_model_last.h5')
