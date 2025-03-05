import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the trained model
model = load_model('food_comparator_model_last.h5')

# Function to preprocess image
def preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
    return img

# Function to predict similarity between two images
def predict_similarity(img1, img2):
    img1 = preprocess_image(img1)
    img2 = preprocess_image(img2)
    prediction = model.predict([np.expand_dims(img1, axis=0), np.expand_dims(img2, axis=0)])
    return prediction[0][0]

# Read the CSV file with image paths and winner column
df = pd.read_csv('test.csv')

# Folder where images are stored
image_folder = 'Test/'

# Iterate through the rows of the dataframe
for index, row in df.iterrows():
    img1_path = image_folder + row['Image 1']  # Use correct column name
    img2_path = image_folder + row['Image 2']  # Use correct column name

    # Get the similarity score
    similarity = predict_similarity(img1_path, img2_path)
    print(f"Similarity score for row {index}: {similarity:.4f}")

    # Show both images and the result
    img1 = load_img(img1_path)
    img2 = load_img(img2_path)

    # Plot the images
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Display image 1
    ax[0].imshow(img1)
    ax[0].axis('off')
    ax[0].set_title('Image 1')

    # Display image 2
    ax[1].imshow(img2)
    ax[1].axis('off')
    ax[1].set_title('Image 2')

    # Display result
    plt.suptitle(f"Similarity score: {similarity:.4f}\n"
                 f"{'The images are similar (Image 1 is better)' if similarity > 0.5 else 'The images are different (Image 2 is better)'}", fontsize=14)

    plt.show()

    # Update the winner column based on the similarity score
    if similarity > 0.5:
        df.at[index, 'Winner'] = 1  # Image 1 is better
    else:
        df.at[index, 'Winner'] = 2  # Image 2 is better

# Save the updated dataframe to a new CSV file
df.to_csv('test_updated.csv', index=False)

print("CSV file updated and saved as 'test_updated.csv'")
