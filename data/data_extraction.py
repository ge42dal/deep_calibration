import pandas as pd
import numpy as np
import zipfile
import io
import requests 
from pathlib import Path
from datetime import date
from datetime import timedelta


def daterange(start_date: date, end_date: date):
    days = int((end_date - start_date).days)
    for n in range(days):
        yield start_date + timedelta(n)

def pull_data_binance():
    start_date =date(2023,9,8)
    end_date = date(2023,9,18)
    for day in daterange(start_date, end_date):
        date_str = day.strftime("%Y-%m-%d")
        url = f'https://data.binance.vision/data/option/daily/EOHSummary/BTCUSDT/BTCUSDT-EOHSummary-{date_str}.zip'
        try:
            r = requests.get(url)
            r.raise_for_status() 
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for {day}: {e}")
            continue
        z = zipfile.ZipFile(io.BytesIO(r.content))
        filelist = z.namelist()
        z.extractall(str(Path(__file__).parent))


    
    
    filelist = z.namelist()
    data = pd.read_csv(filelist[0])
    print(data.head())





def extract_data_csv_to_pandas(file_path):
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
    # data_binance_vol = extract_data(str(Path(__file__).parent) + '/BTCBVOLUSDT-BVOLIndex-2025-04-30.csv')
    # data_binance_eoh = extract_data(str(Path(__file__).parent) + '/BTCUSDT-EOHSummary-2023-10-23.csv')
    # print(data_binance_eoh.head())
    # print(data_binance_eoh.describe())

    # https://data.binance.vision/data/option/daily/EOHSummary/BTCUSDT/BTCUSDT-EOHSummary-2023-10-23.zip
    pull_data_binance()


    


