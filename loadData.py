import pandas as pd

class loadData():
    def load_tabular_data(base_dir, local_path):
        """
        Input: Base directory and the path where csv file is present
        Output: Returns dataframe
        Function: Loads the tabular data in tab_data.csv
        """
        train_csv = pd.read_csv(base_dir+local_path+'tab_data.csv')
        return train_csv
    
    def load_image_data(base_dir, local_path, csv):
        """
        Input: Base directory and the path to image folder with dataframe
        Output: List of all the images relative path
        Function: Loads all the images from the images folder
        """
        return (f'{base_dir}{local_path}images/'+csv.loc[:,"id"].astype(str)+'.jpeg').values
