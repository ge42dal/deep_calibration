import pandas as pd 
from pathlib import Path
import os

def merge_data(path:str):

    csv_directory = path
    csv_files = [file for file in os.listdir(csv_directory) if file.endswith('.csv')]

    dfs = []

    # Iterate over each CSV file and read it into a DataFrame
    for csv_file in csv_files:
        file_path = os.path.join(csv_directory, csv_file)
        df = pd.read_csv(file_path)
        dfs.append(df)

    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_csv('merged_data', index=False)

if __name__ == "__main__":
    merge_data(str(Path(__file__).parent.parent) + '/data')


