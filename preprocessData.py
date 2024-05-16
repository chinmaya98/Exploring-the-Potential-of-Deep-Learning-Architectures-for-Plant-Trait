import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler

class preprocessData():
    def pre_process_data(train_csv):
        TARGET = ['X4_mean', 'X11_mean', 'X18_mean','X26_mean', 'X50_mean', 'X3112_mean']
        Unnec = ['X4_mean', 'X11_mean', 'X18_mean','X26_mean', 'X50_mean', 'X3112_mean',
                'X4_sd', 'X11_sd', 'X18_sd','X26_sd', 'X50_sd', 'X3112_sd', 'id']
        
        target = train_csv[TARGET]
        train_csv = train_csv.drop(Unnec, axis=1)

        train_csv = (train_csv - train_csv.mean()) / train_csv.std()

        train_labels= np.arcsinh(target.values)

        return train_csv, train_labels

    def adjust_image_attributes(image_features, target=None):
        INPUT_SHAPE = (512, 512, 3)
        image_data = tf.io.read_file(image_features[0])
        image_data = tf.io.decode_jpeg(image_data)
        image_data = tf.image.resize(image_data, (INPUT_SHAPE[0], INPUT_SHAPE[1]))
        image_data = image_data / 255.0
    
        if target is None:
            return (image_data, image_features[1])
        else:
            return ((image_data, image_features[1]), target)
    
    def split_data(train_img_path, train_csv, train_labels):
        train_tensor_dataset = tf.data.Dataset.from_tensor_slices(((train_img_path,
                                                                    train_csv) , 
                                                                    train_labels))

        train_size = int(len(train_tensor_dataset) * 0.6)
        val_size = int(len(train_tensor_dataset) * 0.2)
        test_size = len(train_tensor_dataset) - train_size - val_size

        train_csv = train_tensor_dataset.take(train_size)
        remaining_dataset = train_tensor_dataset.skip(train_size)
        val_dataset = remaining_dataset.take(val_size)
        test_csv = remaining_dataset.skip(test_size)

        train_data = train_csv.map(preprocessData.adjust_image_attributes).shuffle(buffer_size=256).batch(32).prefetch(16)
        val_data = val_dataset.map(preprocessData.adjust_image_attributes).batch(32).prefetch(16)
        test_data = test_csv.map(preprocessData.adjust_image_attributes).batch(32).prefetch(16)

        return train_data, val_data, test_data


        
