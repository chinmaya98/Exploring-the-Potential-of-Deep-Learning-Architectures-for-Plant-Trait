import os

from setupGPUS import setupGPUS
from loadData import loadData
from preprocessData import preprocessData
from buildModel import custModel
from visualization import visualization

if __name__ ==  '__main__':
    setupGPUS.check_GPUS()

    base_dir = os.getcwd()
    local_path = "/data/planttraits2024/"

    train_csv = loadData.load_tabular_data(base_dir, local_path)
    train_img_path = loadData.load_image_data(base_dir, train_csv, local_path)

    train_csv, train_labels = preprocessData.pre_process_data(train_csv)

    INPUT_SHAPE=(512, 512, 3)
    train_data, val_data, test_data = preprocessData.split_data(train_img_path, 
                                                                train_csv, 
                                                                train_labels)

    model = custModel.build(INPUT_SHAPE, 
                            train_csv, 
                            train_labels)

    history = custModel.compile_fit(model, 
                                    train_data, 
                                    val_data)

    visualization.mae_plot(history)
    visualization.r2_plot(history)

    custModel.test_evaluation(model, test_data)


