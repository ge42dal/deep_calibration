import pandas as pd
import gzip
import numpy as np
from pathlib import Path

def extract_data(file_path):
    """
    Extracts data from a CSV file and returns it as a pandas DataFrame.
    
    Args:
        file_path (str): The path to the CSV file.
        
    Returns:
        pd.DataFrame: The extracted data.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
        return None
    except pd.errors.ParserError:
        print("Error: The file could not be parsed.")
        return None
if __name__ == "__main__":
    data_binance_vol = extract_data(str(Path(__file__).parent) + '/BTCBVOLUSDT-BVOLIndex-2025-04-30.csv')
    data_binance_eoh = extract_data(str(Path(__file__).parent) +'/BTCUSDT-EOHSummary-2023-10-23.csv')
    print(data_binance_vol.columns)
    print(data_binance_eoh.columns)


    


