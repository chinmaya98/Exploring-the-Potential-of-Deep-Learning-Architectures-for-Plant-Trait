import pandas as pd

class loadData():
    def load_tabular_data(base_dir, local_path):
        train_csv = pd.read_csv(base_dir+local_path+'train.csv')
        return train_csv
    
    def load_image_data(base_dir, train_csv, local_path):
        return (f'{base_dir}{local_path}train_images/'+train_csv.loc[:,"id"].astype(str)+'.jpeg').values


