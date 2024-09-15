# KaggleCompetition01
This is the answer to the question https://www.kaggle.com/competitions/planttraits2024/overview

Exploring the Potential of Deep Learning Architectures for Plant Trait Prediction Dataset The dataset was taken from Kaggle which is now remove so adding the data in google drive and sharing link. The data is 3.17GB which has one directory named images and on csv file named tab_data.csv. Images has names which is referenced in csv file as id. Download the data and place it in a directory named “data”. The structure should be as shown below once the data (images/* and tab_data.csv is downloaded into data). DATA255_Group2_Project |─ code | ├── buildModel.py | ├── loadData.py | ├── main.py | ├── preprocessData.py | ├── setupGPUS.py | └── visualization.py |─ data | ├── images | └── tab_data.csv |─ readme.docx |─ requirements.txt Steps to run the code: Prerequisites: Conda and Linux/MacOS Once the data is downloaded and placed in the “data” folder,

Open the terminal in the unzipped folder and create a new environment using conda conda create -n env_name python
Activate the environment using conda activate env_name
Install all the libraries mentioned in requirements.txt from the conda environment pip install -r requirements.txt
Once all the libraries are installed go into code using cd code
And run python main.py Following the above steps will run the main.py and modelling will start and run for 50 epochs giving the results and showing loss curve with evaluation metric. Running this in HPC lab each epoch took 40s so for the program to run in whole it would take ~40mins including load, preprocess and modelling.
