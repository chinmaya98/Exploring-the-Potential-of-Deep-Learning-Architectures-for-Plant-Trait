import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler

class preprocessData():
    def pre_process_data(train_csv):
        Target_variables = ['X4_mean', 'X11_mean', 'X18_mean','X26_mean', 'X50_mean', 'X3112_mean']
        train_csv = train_csv.dropna(axis=1)

        features= train_csv.drop('id',axis=1).columns

        scaler = StandardScaler()
        train_feature = scaler.fit_transform(train_csv[features].values)

        train_labels= np.arcsinh(train_csv[Target_variables].values)

        return train_feature, train_labels, features

    def adjust_image_attributes(image_features, target=None):
        INPUT_SHAPE = (512, 512, 3)
        # Load the image from the file path
        image_data = tf.io.read_file(image_features[0])
        image_data = tf.io.decode_jpeg(image_data)
        image_data = tf.image.resize(image_data, (INPUT_SHAPE[0], INPUT_SHAPE[1]))
        image_data = image_data / 255.0  # Normalize pixel values to the range 0-1
    
        # The function returns a tuple with image and additional features, optionally including the target
        if target is None:
            return (image_data, image_features[1])  # Return data without target for prediction scenarios
        else:
            return ((image_data, image_features[1]), target)  # Include target for training scenarios
    
    def split_data(train_img_path, feature, labels, INPUT_SHAPE):
        ## create tensorflow dataset 
        train_tensor_dataset = tf.data.Dataset.from_tensor_slices(((train_img_path,
                                                                    feature) , 
                                                                    labels))

        # Define sizes for train, validation, and test sets
        train_size = int(len(train_tensor_dataset) * 0.6)
        val_size = int(len(train_tensor_dataset) * 0.2)
        test_size = len(train_tensor_dataset) - train_size - val_size

        # Split the dataset into train, validation, and test sets
        train_csv = train_tensor_dataset.take(train_size)
        remaining_dataset = train_tensor_dataset.skip(train_size)
        val_dataset = remaining_dataset.take(val_size)
        test_csv = remaining_dataset.skip(test_size)

        # Apply preprocessing and batching to train, validation, and test datasets
        train_data = train_csv.map(preprocessData.adjust_image_attributes).shuffle(buffer_size=256).batch(32).prefetch(16)
        val_data = val_dataset.map(preprocessData.adjust_image_attributes).batch(32).prefetch(16)
        test_data = test_csv.map(preprocessData.adjust_image_attributes).batch(32).prefetch(16)

        return train_data, val_data, test_data


        
