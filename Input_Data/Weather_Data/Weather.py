import requests
from bs4 import BeautifulSoup
import pandas as pd
import zipfile
from io import BytesIO
from urllib.parse import urljoin
import logging
from pathlib import Path
from typing import List
from Settings.config import getPaths
import glob
import openmeteo_requests
import requests_cache
from retry_requests import retry


class WeatherDataDownloader:
    def __init__(self, stations: List[str], base_urls: List[str], output_dir: str):
        """
        Initialize the downloader with base URLs and output directory.

        Args:
            stations
            base_urls (List[str]): List of base URLs to fetch weather data from
            output_dir (str): Path to the output directory where files will be saved
        """
        self.stations = stations
        self.base_urls = base_urls

        # Convert output_dir to Path object and resolve to absolute path
        self.output_dir = Path(output_dir).resolve()

        # Create necessary subdirectories
        self.parquet_dir = self.output_dir / 'parquet'
        self.temp_dir = self.output_dir / 'temp'

        # Create directories if they don't exist
        self._setup_directories()

        logging.info(f"Initialized downloader with output directory: {self.output_dir}")
        logging.info(f"Number of base URLs to process: {len(self.base_urls)}")

    def _setup_directories(self):
        """Create necessary directory structure and validate paths."""
        try:
            # Create main output directory
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Create subdirectories
            self.parquet_dir.mkdir(exist_ok=True)
            self.temp_dir.mkdir(exist_ok=True)

            # Verify write permissions
            test_file = self.temp_dir / 'test_write.tmp'
            test_file.touch()
            test_file.unlink()

        except PermissionError:
            raise PermissionError(f"No write permission in directory: {self.output_dir}")
        except Exception as e:
            raise Exception(f"Error setting up directories: {e}")

    def get_file_list(self, base_url: str) -> List[str]:
        """
        Fetch list of all zip files from a specific URL.

        Args:
            stations
            base_url (str): The URL to fetch files from

        Returns:
            List[str]: List of complete URLs to zip files
        """
        try:
            response = requests.get(base_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find all links that end with .zip
            links = [link.get('href') for link in soup.find_all('a')
                     if link.get('href', '').endswith('.zip') and
                     any(f in link.get('href', '') for f in self.stations)]

            return [urljoin(base_url, link) for link in links]

        except requests.RequestException as e:
            logging.error(f"Error fetching file list from {base_url}: {e}")
            return []

    def download_and_process_file(self, url: str):
        """Download, unzip, and process a single file."""
        try:
            filename = url.split('/')[-1]
            station_id = filename.split('_')[2]

            # Get the data type from the URL (e.g., 'air_temperature', 'precipitation')
            data_type = url.split('/')[-3]
            parquet_path = self.parquet_dir / f"{data_type}_{station_id}.parquet"

            # Skip if parquet file already exists
            if parquet_path.exists():
                logging.info(f"Skipping {filename} - already processed")
                return

            logging.info(f"Downloading {filename}")
            response = requests.get(url)
            response.raise_for_status()

            # Read zip file from memory
            with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
                # Extract to temporary directory
                zip_ref.extractall(self.temp_dir)

                # Process each file in the zip
                for extracted_file in zip_ref.namelist():
                    if extracted_file.startswith('produkt_') and extracted_file.endswith('.txt'):
                        file_path = self.temp_dir / extracted_file

                        # Read and process the data with low_memory=False to avoid warnings
                        df = pd.read_csv(
                            file_path,
                            sep=';',
                            low_memory=False,
                            na_values=[-999, -999.0]  # Specify all possible NA values
                        )

                        # Basic data cleaning
                        df = df.replace(-999, pd.NA)

                        # Add metadata columns
                        df['data_type'] = data_type
                        df['station_id'] = station_id
                        df['source_file'] = filename

                        # Save as parquet
                        df.to_parquet(parquet_path, index=False)

                        # Clean up temporary file
                        file_path.unlink()

            logging.info(f"Successfully processed {filename}")

        except Exception as e:
            logging.error(f"Error processing {url}: {e}")

    def process_all_files(self):
        """Download and process all files from all base URLs."""
        for base_url in self.base_urls:
            logging.info(f"Processing files from: {base_url}")
            files = self.get_file_list(base_url)
            logging.info(f"Found {len(files)} files to process from {base_url}")

            for file_url in files:
                self.download_and_process_file(file_url)


def getWeather(df_stations, historical=False):
    if historical:
        actual = 'historical'
    else:
        actual = 'recent'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    stations = df_stations['Stations_id'].unique().tolist()
    baseUrl = [
        getPaths('tempURL') + f'{actual}/',
        getPaths('cloudURL') + f'{actual}/',
        getPaths('cloudinessURL') + f'{actual}/',
        getPaths('extremeWindURL') + f'{actual}/',
        getPaths('moistureURL') + f'{actual}/',
        getPaths('precipitationURL') + f'{actual}/',
        getPaths('pressureURL') + f'{actual}/',
        getPaths('soilTempURL') + f'{actual}/',
        getPaths('solarURL') + f'{actual}/',
        getPaths('sunURL') + f'{actual}/',
        getPaths('visibilityURL') + f'{actual}/',
        getPaths('weatherPhenomenaURL') + f'{actual}/',
        getPaths('windURL') + f'{actual}/',
        getPaths('windSynopURL') + f'{actual}/',
    ]
    try:
        downloader = WeatherDataDownloader(stations, baseUrl, getPaths('outputWeatherData')+f'{actual}')
        downloader.process_all_files()
    except Exception as e:
        logging.error(f"Failed to initialize downloader: {e}")
        raise


def combineData(historical=False):
    if historical:
        actual = 'historical'
    else:
        actual = 'recent'
    data_types = [
          'air_temperature',
          'cloud_type',
          'cloudiness',
          'extreme_wind',
          'moisture',
          'precipitation',
          'pressure',
          'soil_temperature',
          'sun',
          'visibility',
          'wind',
          'wind_synop'
                  ]
    for data_type in data_types:
        pattern = f"{getPaths("outputWeatherData")}{actual}/parquet/{data_type}_*.parquet"
        parquet_files = glob.glob(pattern)

        # if not parquet_files:
        #     logging.warning(f"No files found for {data_type} in {getPaths("outputWeatherData")}")
        #     return

        dfs = []
        for file in parquet_files:
            try:
                df = pd.read_parquet(file)
                dfs.append(df)
            except Exception as e:
                logging.error(f"Error reading {file}: {e}")
                continue

        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            output_file = f"{getPaths("outputWeatherData")}{actual}/{data_type}_combined.parquet"
            combined_df.to_parquet(output_file, index=False)
            logging.info(f"Saved combined file to {output_file}")


def combineHistoricRecent():
    data_types = [
        'air_temperature',
        'cloud_type',
        'cloudiness',
        'extreme_wind',
        'moisture',
        'precipitation',
        'pressure',
        'soil_temperature',
        'sun',
        'visibility',
        'wind',
        'wind_synop'
    ]

    for data_type in data_types:
        dfOld = pd.read_parquet(f"{getPaths("outputWeatherData")}historical/{data_type}_combined.parquet")
        dfNew = pd.read_parquet(f"{getPaths("outputWeatherData")}recent/{data_type}_combined.parquet")
        df = pd.concat([dfOld, dfNew], ignore_index=True)
        df['MESS_DATUM'] = df['MESS_DATUM'].astype(str)
        df['date'] = pd.to_datetime(df['MESS_DATUM'].str[:8], format='%Y%m%d')
        df = df[df['date'] >= '2015-01-01'].reset_index(drop=True)
        df.to_parquet(f"{getPaths("outputWeatherData")}combined/{data_type}_combined.parquet")


def forecastMeteo(df_stations):

    # Set up caching and retry mechanisms
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    df_weatherforecast = pd.DataFrame()
    for index, row in df_stations.iterrows():
        latitude = row['geoBreite']/10000
        longitude = row['geoLaenge']/10000
        # Define API URL and parameters
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation", "sunshine_duration"],
            "timezone": "Europe/Berlin",
            "past_days": 2
        }

        # Send the request to Open-Meteo API
        responses = openmeteo.weather_api(url, params=params)

        # Process the first location in the response (add a loop if there are multiple locations)
        response = responses[0]
        print(f"Coordinates: {response.Latitude()}°N, {response.Longitude()}°E")
        print(f"Elevation: {response.Elevation()} m asl")
        print(f"Timezone: {response.Timezone()} ({response.TimezoneAbbreviation()})")
        print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()} s")

        # Process hourly data
        hourly = response.Hourly()
        hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
        hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
        hourly_wind_speed_10m = hourly.Variables(2).ValuesAsNumpy()
        precipitation = hourly.Variables(3).ValuesAsNumpy()
        sunshine_duration = hourly.Variables(4).ValuesAsNumpy()

        # Construct a DataFrame with hourly data
        hourly_data = {
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=False),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=False),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            ),
            "temperature_2m": hourly_temperature_2m,
            "relative_humidity_2m": hourly_relative_humidity_2m,
            "wind_speed_10m": hourly_wind_speed_10m,
            "precipitation": precipitation,
            "sunshine_duration": sunshine_duration
        }

        hourly_dataframe = pd.DataFrame(data=hourly_data)
        hourly_dataframe['stationsID'] = row['Stations_id']
        df_weatherforecast = pd.concat([df_weatherforecast, hourly_dataframe])

    return df_weatherforecast
