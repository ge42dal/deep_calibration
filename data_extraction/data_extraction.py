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


def pull_data_binance(start_date: date, end_date: date):
    """
    Pulls data from Binance for the specified date range.
    Args:
        start_date (date): Start date for data extraction.
        end_date (date): End date for data extraction.
    """
     
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
        z.extractall(str(Path(__file__).parent))


if __name__ == "__main__":
    start_date = date(2023, 9, 8)
    end_date = date(2023, 9, 18)
    # pull_data_binance(start_date, end_date)
   