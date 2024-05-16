import os
import keras
import numpy as np 
import pandas as pd 
import tensorflow as tf 
import matplotlib.pyplot as plt

from tensorflow.keras import backend as K


from setupGPUS import setupGPUS
from loadData import loadData
from preprocessData import preprocessData
from buildModel import evaluationMetrics
from buildModel import custModel
from visualization import visualization

import tensorflow as tf


setupGPUS.check_GPUS()

base_dir = os.getcwd()
local_path = "data/planttraits2024/"

train_csv = loadData.load_tabular_data(base_dir, local_path)
train_img_path = loadData.load_image_data(base_dir, train_csv)

train_feature, train_labels, features = preprocessData.pre_process_data(train_csv)

INPUT_SHAPE=(512, 512, 3)
train_data, val_data, test_data = preprocessData.split_data(train_img_path, 
                                                            train_feature, 
                                                            train_labels, 
                                                            INPUT_SHAPE)

model = custModel.build(INPUT_SHAPE, features)

history = custModel.compile_fit(model, 
                            train_data, 
                            val_data)

visualization.mae_loss(history)
# visualization.r2(history)

# evaluationMetrics.r2(y_true, y_pred)
# evaluationMetrics.RMSE(y_true, y_pred)
# evaluationMetrics.MAE(y_true, y_pred)

custModel.test_evaluation(model, test_data)


