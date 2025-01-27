from Preprocessing.preprocessing import *
import Evaluation.models_eval as evaluation
from Settings.input import *
from functions_baseline import *

df_energy = cleanEnergy(importEnergy())

df_bike_daily = importBike()
df_bike_daily = preprocessBike(df_bike_daily)
df_bike_daily = remove_outliers(df_bike_daily, column='bike_count')

save_dir = getPaths('baseline_save')
energy_dir = os.path.join(save_dir, 'energy')
bike_daily_dir = os.path.join(save_dir, 'bike_daily')

print(df_bike_daily.columns)
print(df_bike_daily.dtypes)
print(df_bike_daily.head())

print(df_energy.columns)
print(df_energy.dtypes)
print(df_energy.head())


forecast_start = '2024-01-01 00:00'
forecast_end = '2024-12-31 23:00'

for dir_path in [save_dir, energy_dir, bike_daily_dir]:
    os.makedirs(dir_path, exist_ok=True)


energy_results, bike_results = run_evaluation(
    df_energy,
    df_bike_daily,
    save_dir,
    forecast_start,
    forecast_end
)



