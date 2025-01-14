import pandas as pd
import tensorflow as tf

# Define the path to the images
image_path = "/Users/manojreddy/Documents/Comparable/Data/Images/"

# Function to load the dataset and prepare image data
def load_scut(file="/Users/manojreddy/Documents/Comparable/Data/ImageExp/All_Ratings3.csv"):
    """
    Load the art dataset from the given CSV file and preprocess image data.
    
    :param file: Path to the CSV file (either All_Ratings3.csv or Selected_Ratings3.csv)
    :return: DataFrame with image paths and processed pixel values
    """
    
    def retrievePixels(path):
        """
        Helper function to load and preprocess images.
        
        :param path: Path to the image file
        :return: Processed image as a numpy array
        """
        img = tf.keras.utils.load_img(image_path + path, target_size=(250, 250))
        x = tf.keras.utils.img_to_array(img)
        return x

    # Load the dataset from the specified CSV file
    data = pd.read_csv(file)
    
    # Assuming the 'Filename' column holds the image file names, apply image preprocessing
    data['pixels'] = data['Filename'].apply(retrievePixels)
    
    # Return the DataFrame with images converted to pixel arrays
    return data


# Example usage:
# Loading All_Ratings3.csv for further use
all_ratings_data = load_scut("/Users/manojreddy/Documents/Comparable/Data/ImageExp/All_Ratings3.csv")
print(all_ratings_data.head())  # Print the first few rows of the loaded dataset
