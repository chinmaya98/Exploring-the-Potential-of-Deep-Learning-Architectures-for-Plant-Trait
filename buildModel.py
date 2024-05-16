import tensorflow as tf
from tensorflow.keras import backend as K

from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D,Input,concatenate 
from tensorflow.keras.applications import MobileNetV2

class evaluationMetrics():
    def r2(y_true, y_predict):
        y_true = tf.cast(y_true, tf.float32)
        y_predict = tf.cast(y_predict, tf.float32)
        SS_res  = K.sum(K.square(y_true - y_predict))
        SS_tot = K.sum(K.square(y_true- K.mean(y_true)))
        return 1- (SS_res/(SS_tot+ K.epsilon()))
    
    def RMSE(y_true, y_predict):
        y_true = tf.cast(y_true, tf.float32)
        y_predict = tf.cast(y_predict, tf.float32)
        error = y_predict - y_true
        squared_error = tf.square(error)
        mean_squared_error = tf.reduce_mean(squared_error)
        return tf.sqrt(mean_squared_error)
    
    def MAE(y_true, y_predict):
        y_true = tf.cast(y_true, tf.float32)
        y_predict = tf.cast(y_predict, tf.float32)
        error = tf.abs(y_predict - y_true)
        return tf.reduce_mean(error)


class custModel():
    def build(INPUT_SHAPE, features):
        # Load MobileNet as the base model for image processing
        image_base = MobileNetV2(weights='imagenet', include_top=False)

        # Create image input path
        image_input = Input(shape=INPUT_SHAPE, name='image_input')
        x_image = image_base(image_input)
        x_image = GlobalAveragePooling2D()(x_image)

        # Create numerical input path
        numeric_input = Input(shape=(len(features),), name='numeric_input')  # Adjust 'len(feature_cols)' as needed
        x_numeric = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(numeric_input)

        # Concatenate image and numerical features
        combined_features = concatenate([x_image, x_numeric], axis=-1)

        # Add a few more layers with L2 regularization
        x = Dense(1024, activation='relu', kernel_regularizer=l2(0.01))(combined_features)
        x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)

        # Output layer
        output = Dense(6, activation='linear')(x)  

        # Create model
        model = Model(inputs=[image_input, numeric_input], outputs=output)

        # Summary of the model
        model.summary()

        return model

    def compile_fit(model, train_data, val_data):
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='MAE',  
            metrics=[evaluationMetrics.RMSE])
        
        history = model.fit(train_data,
                            epochs=50,
                            batch_size=512,
                            validation_data=val_data)
        
        return history
    
    def test_evaluation(model, test_data):
        # Evaluate the model and get the R^2 score
        results = model.evaluate(test_data)
        r2 = results[1]

        print(r2)

