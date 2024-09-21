from sklearn.metrics import accuracy_score
import pandas as pd
import os

additional_data_folder = 'C:/Users/narasimha.chandi/OneDrive - HCL TECHNOLOGIES LIMITED/Desktop/Testing/RetrainFolder'

def calculate_accuracy(folder_path):
    dataframes = []
    for filename in os.listdir(additional_data_folder):
        if filename.endswith('.csv'): 
            file_path = os.path.join(additional_data_folder, filename)
            file_path = file_path.replace("\\", "/")
            files = pd.read_csv(file_path)  
            dataframes.append(files)
        
    df = pd.concat(dataframes, ignore_index=True)
    df.drop_duplicates(keep='first', inplace=True)
    accuracy = accuracy_score(df['Label'], df['Prediction'])
    return accuracy*100

accuracy = calculate_accuracy(additional_data_folder)
print(f"Accuracy of the model: {accuracy} %")