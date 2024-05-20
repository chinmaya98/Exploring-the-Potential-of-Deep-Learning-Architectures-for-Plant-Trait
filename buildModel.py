import tensorflow as tf
from tensorflow.keras import backend as K

from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D,Input,concatenate 
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.callbacks import EarlyStopping

class evaluationMetrics():
    def r2(y_true, y_predict):
        """
        Input: True and predicted values
        Output: R2 Score
        """
        y_true = tf.cast(y_true, tf.float32)
        y_predict = tf.cast(y_predict, tf.float32)
        SS_res  = K.sum(K.square(y_true - y_predict))
        SS_tot = K.sum(K.square(y_true- K.mean(y_true)))
        return 1- (SS_res/(SS_tot+ K.epsilon()))
    
    def RMSE(y_true, y_predict):
        """
        Input: True and predicted values
        Output: RMSE Loss Value
        """
        y_true = tf.cast(y_true, tf.float32)
        y_predict = tf.cast(y_predict, tf.float32)
        error = y_predict - y_true
        squared_error = tf.square(error)
        mean_squared_error = tf.reduce_mean(squared_error)
        return tf.sqrt(mean_squared_error)
    
    def MAE(y_true, y_predict):
        """
        Input: True and predicted values
        Output: MAE Loss Value
        """
        y_true = tf.cast(y_true, tf.float32)
        y_predict = tf.cast(y_predict, tf.float32)
        error = tf.abs(y_predict - y_true)
        return tf.reduce_mean(error)


class custModel():
    def build(INPUT_SHAPE, features):
        """
        Input: INPUT SHAPE of image and tab features
        Output: Defines the architectue of the model
        Function: Builds the architecture and returns model
        """
        image_base = MobileNetV2(weights='imagenet', include_top=False)

        image_input = Input(shape=INPUT_SHAPE, name='image_input')
        x_image = image_base(image_input)
        x_image = GlobalAveragePooling2D()(x_image)

        numeric_input = Input(shape=(len(features),), name='numeric_input')
        x_numeric = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(numeric_input)

        combined_features = concatenate([x_image, x_numeric], axis=-1)

        x = Dense(1024, activation='relu', kernel_regularizer=l2(0.01))(combined_features)
        x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)

        output = Dense(6, activation='linear')(x)  


        model = Model(inputs=[image_input, numeric_input], outputs=output)

        model.summary()

        return model

    def compile_fit(model, train_data, val_data):
        """
        Input: Model obj, train data and validation data
        Output: History of the built and fitted model
        Function: Compiles the model and fits the model 
        """
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                      loss=MeanAbsoluteError(), 
                      metrics=[evaluationMetrics.r2, evaluationMetrics.RMSE])
        
        early_stopper = EarlyStopping(monitor='val_r_squared',
                                      mode='max',
                                      min_delta=0.001,
                                      patience=10,
                                      verbose=1,
                                      restore_best_weights=True)
        
        history = model.fit(train_data,
                            epochs=50,
                            batch_size=32,
                            validation_data=val_data,
                            callbacks=[early_stopper])
        
        return history
    
    def test_evaluation(model, test_data):
        """
        Input: Model object and test data
        Output: r2 score for test data on model
        Function: Calucate the R2 score for test data
        """
        results = model.evaluate(test_data)
        r2 = results[1]

        print(r2)

