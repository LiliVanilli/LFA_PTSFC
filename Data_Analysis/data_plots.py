from Settings.input import *
from Preprocessing.preprocessing import *
from Features.Bike.bike_daily_features import *
from Features.Bike.bike_hourly_features import *
from Features.Energy.energy_features import *
from Evaluation.models_eval import *
import pandas as pd
from matplotlib import pyplot as plt


def plot_seasonal_patterns_kit(df, y_column, y_label, title_base, save_path=None):
    try:
        # KIT Colors (RGB)
        KIT_COLORS = {
            'gruen': (0 / 255, 150 / 255, 130 / 255),  # Main green
            'blau': (70 / 255, 100 / 255, 170 / 255),  # Blue
            'schwarz': (64 / 255, 64 / 255, 64 / 255),  # 70% Black
            'maigruen': (140 / 255, 182 / 255, 60 / 255),  # May green
            'gelb': (252 / 255, 229 / 255, 0 / 255),  # Yellow
            'orange': (223 / 255, 155 / 255, 27 / 255)  # Orange
        }

        TITLE_SIZE = 20
        LABEL_SIZE = 20
        TICK_SIZE = 20
        LEGEND_SIZE = 20

        # Ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(df['dateTime']):
            df['dateTime'] = pd.to_datetime(df['dateTime'])

        # Create copy of dataframe with additional time features
        analysis_df = df.copy()
        analysis_df['month'] = analysis_df['dateTime'].dt.month
        analysis_df['hour'] = analysis_df['dateTime'].dt.hour
        analysis_df['dayofweek'] = analysis_df['dateTime'].dt.dayofweek

        def get_season(month):
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Spring'
            elif month in [6, 7, 8]:
                return 'Summer'
            else:  # 9, 10, 11
                return 'Fall'

        analysis_df['season'] = analysis_df['month'].apply(get_season)

        # Plot 1: Monthly Pattern
        plt.figure(figsize=(15, 6))
        monthly_stats = analysis_df.groupby('month')[y_column].agg(['mean', 'std']).reset_index()

        plt.plot(monthly_stats['month'], monthly_stats['mean'],
                 color=KIT_COLORS['gruen'], linewidth=2, marker='o')

        plt.fill_between(monthly_stats['month'],
                         monthly_stats['mean'] - monthly_stats['std'],
                         monthly_stats['mean'] + monthly_stats['std'],
                         color=KIT_COLORS['gruen'], alpha=0.2)

        plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize=TICK_SIZE)
        plt.title(f'{title_base}: Monthly Pattern', fontsize=TITLE_SIZE)
        plt.xlabel('Month', fontsize=LABEL_SIZE)
        plt.ylabel(y_label, fontsize=LABEL_SIZE)
        plt.legend(fontsize=LEGEND_SIZE)
        plt.grid(True, linestyle='--', alpha=0.7, color=KIT_COLORS['schwarz'])

        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            monthly_path = os.path.join(save_path, f"{title_base.replace(' ', '_').lower()}_monthly.png")
            plt.savefig(monthly_path, dpi=300, bbox_inches='tight')
            print(f"Monthly plot saved to {monthly_path}")

        plt.close()

        # Plot 2: Daily Pattern by Season
        plt.figure(figsize=(15, 6))

        season_colors = {
            'Winter': KIT_COLORS['blau'],
            'Spring': KIT_COLORS['maigruen'],
            'Summer': KIT_COLORS['gelb'],
            'Fall': KIT_COLORS['orange']
        }

        for season in ['Winter', 'Spring', 'Summer', 'Fall']:
            season_data = analysis_df[analysis_df['season'] == season]
            hourly_avg = season_data.groupby('hour')[y_column].mean()

            plt.plot(hourly_avg.index, hourly_avg.values,
                     label=season, color=season_colors[season],
                     linewidth=2)

        plt.xticks(range(0, 24, 2), fontsize=TICK_SIZE)
        plt.xlabel('Hour of Day', fontsize=LABEL_SIZE)
        plt.ylabel(y_label, fontsize=LABEL_SIZE)
        plt.title(f'{title_base}: Daily Patterns by Season', fontsize=TITLE_SIZE)
        plt.legend(fontsize=LEGEND_SIZE)
        plt.grid(True, linestyle='--', alpha=0.7, color=KIT_COLORS['schwarz'])

        if save_path:
            daily_path = os.path.join(save_path, f"{title_base.replace(' ', '_').lower()}_daily.png")
            plt.savefig(daily_path, dpi=300, bbox_inches='tight')
            print(f"Daily plot saved to {daily_path}")

        plt.close()

        # Plot 3: Weekly Pattern by Season
        plt.figure(figsize=(15, 6))

        for season in ['Winter', 'Spring', 'Summer', 'Fall']:
            season_data = analysis_df[analysis_df['season'] == season]
            weekly_avg = season_data.groupby('dayofweek')[y_column].mean()

            plt.plot(weekly_avg.index, weekly_avg.values,
                     label=season, color=season_colors[season],
                     linewidth=2, marker='o')

        plt.xticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], fontsize=TICK_SIZE)
        plt.xlabel('Day of Week', fontsize=LABEL_SIZE)
        plt.ylabel(y_label, fontsize=LABEL_SIZE)
        plt.title(f'{title_base}: Weekly Patterns by Season', fontsize=TITLE_SIZE)
        plt.legend(fontsize=LEGEND_SIZE)
        plt.grid(True, linestyle='--', alpha=0.7, color=KIT_COLORS['schwarz'])

        if save_path:
            weekly_path = os.path.join(save_path, f"{title_base.replace(' ', '_').lower()}_weekly.png")
            plt.savefig(weekly_path, dpi=300, bbox_inches='tight')
            print(f"Weekly plot saved to {weekly_path}")

        plt.close()

    except Exception as e:
        print(f"Error while creating seasonal pattern plot: {e}")



def scatter_with_trend_bike_daily(df_weather, df_bike_hourly, start_date, end_date, color='green', save_path=None):
    try:
        # Define KIT Grün and KIT Blau colors
        colors = {
            'green': (0 / 255, 150 / 255, 130 / 255),  # KIT Grün
            'blue': (70 / 255, 100 / 255, 170 / 255),  # KIT Blau
        }
        if color not in colors:
            raise ValueError("Invalid color. Choose 'green' or 'blue'.")
        selected_color = colors[color]

        TITLE_SIZE = 20
        LABEL_SIZE = 20
        TICK_SIZE = 20
        LEGEND_SIZE = 20


        # Ensure 'date' in df_weather is datetime
        if not pd.api.types.is_datetime64_any_dtype(df_weather['date']):
            df_weather['date'] = pd.to_datetime(df_weather['date'], errors='coerce')

        # Aggregate weather data to daily average temperature
        daily_weather = df_weather.groupby('date')['avg_temperatur'].mean().reset_index()

        # Ensure 'dateTime' in df_bike_hourly is datetime
        if not pd.api.types.is_datetime64_any_dtype(df_bike_hourly['dateTime']):
            df_bike_hourly['dateTime'] = pd.to_datetime(df_bike_hourly['dateTime'], errors='coerce')

        # Extract date from datetime for bikes data and sum up daily counts
        df_bike_hourly['date'] = df_bike_hourly['dateTime'].dt.date
        df_bike_hourly['date'] = pd.to_datetime(df_bike_hourly['date'])
        daily_bikes = df_bike_hourly.groupby('date')['bike_count'].sum().reset_index()

        # Merge daily weather and bike DataFrames
        merged_data = pd.merge(
            daily_weather,
            daily_bikes,
            on='date',
            how='inner'
        )

        # Filter data within the specified date range
        start_datetime = pd.to_datetime(start_date)
        end_datetime = pd.to_datetime(end_date)
        filtered_data = merged_data[
            (merged_data['date'] >= start_datetime) &
            (merged_data['date'] < end_datetime)
        ]

        if filtered_data.empty:
            raise ValueError("No data available in the specified date range.")

        # Scatter plot
        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            x=filtered_data['avg_temperatur'],
            y=filtered_data['bike_count'],
            color=selected_color,
            alpha=0.5,
            edgecolor='k',
            s=50,  # Slightly larger points for daily data
            label='Daily Values'
        )

        # Add a trend line using a regression fit
        sns.regplot(
            x=filtered_data['avg_temperatur'],
            y=filtered_data['bike_count'],
            scatter=False,
            color="red",
            label='Trend Line'
        )

        # Formatting
        plt.xlabel("Average Daily Temperature (°C)", fontsize=LABEL_SIZE)
        plt.ylabel("Total Daily Bike Count", fontsize=LABEL_SIZE)
        plt.title("Daily Bike Count vs Average Temperature", fontsize=TITLE_SIZE)
        plt.xticks(fontsize=TICK_SIZE)
        plt.yticks(fontsize=TICK_SIZE)
        plt.legend(fontsize=LEGEND_SIZE)
        plt.grid(color='gray', linestyle="--", alpha=0.5)
        plt.tight_layout()

        # Save plot
        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            file_name = "scatter_daily_temp_vs_bike_count_with_trend.png"
            full_path = os.path.join(save_path, file_name)
            plt.savefig(full_path, dpi=300)
            print(f"Plot saved to {full_path}")

        # Show plot
        # plt.show()

    except Exception as e:
        print(f"Error while creating daily scatter plot: {e}")

        # Scatter plot
        plt.scatter(
            filtered_data['avg_temperatur'],
            filtered_data['load'],
            c=[selected_color],
            alpha=0.7,
            edgecolors='k',
            s=50  # Point size
        )

        # Formatting
        plt.xlabel("Temperature (°C)", fontsize=14)
        plt.ylabel("Energy Load (GW)", fontsize=14)
        plt.title("Dependence of Energy Load on Temperature", fontsize=16)
        plt.grid(color='gray', linestyle="--", alpha=0.5)
        plt.tight_layout()

        # Save plot
        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)  # Create directory if it doesn't exist

            file_name = "scatter_temp_vs_load.png"
            full_path = os.path.join(save_path, file_name)
            plt.savefig(full_path, dpi=300)
            print(f"Plot saved to {full_path}")

        # Show plot
        # plt.show()

    except Exception as e:
        print(f"Error while creating scatter plot: {e}")


def scatter_with_trend_energy(df_weather, df_energy, start_date, end_date, color='green', save_path=None):
    try:
        # Define KIT Grün and KIT Blau colors
        colors = {
            'green': (0 / 255, 150 / 255, 130 / 255),  # KIT Grün
            'blue': (70 / 255, 100 / 255, 170 / 255),  # KIT Blau
        }
        if color not in colors:
            raise ValueError("Invalid color. Choose 'green' or 'blue'.")
        selected_color = colors[color]

        TITLE_SIZE = 20
        LABEL_SIZE = 20
        TICK_SIZE = 20
        LEGEND_SIZE = 20

        # Ensure 'date' in df_weather is datetime
        if not pd.api.types.is_datetime64_any_dtype(df_weather['date']):
            df_weather['date'] = pd.to_datetime(df_weather['date'], errors='coerce')

        # Combine 'date' and 'hour' in df_weather to create a proper 'dateTime' column
        df_weather['dateTime'] = pd.to_datetime(df_weather['date']) + pd.to_timedelta(df_weather['hour'], unit='h')

        # Ensure 'dateTime' in df_energy is datetime
        if not pd.api.types.is_datetime64_any_dtype(df_energy['dateTime']):
            df_energy['dateTime'] = pd.to_datetime(df_energy['dateTime'], errors='coerce')

        # Merge weather and energy DataFrames on 'dateTime'
        merged_data = pd.merge(
            df_weather[['dateTime', 'avg_temperatur']],
            df_energy[['dateTime', 'load']],
            on='dateTime',
            how='inner'
        )

        # Filter data within the specified date range
        start_datetime = pd.to_datetime(start_date)
        end_datetime = pd.to_datetime(end_date)
        filtered_data = merged_data[
            (merged_data['dateTime'] >= start_datetime) &
            (merged_data['dateTime'] < end_datetime)
        ]

        if filtered_data.empty:
            raise ValueError("No data available in the specified date range.")

        # Scatter plot
        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            x=filtered_data['avg_temperatur'],
            y=filtered_data['load'],
            color=selected_color,
            alpha=0.5,
            edgecolor='k',
            s=30,
            label='Data Points'
        )

        # Add a trend line using a regression fit
        sns.regplot(
            x=filtered_data['avg_temperatur'],
            y=filtered_data['load'],
            scatter=False,
            color="red",
            label='Trend Line'
        )

        # Formatting
        plt.xlabel("Temperature (°C)", fontsize=LABEL_SIZE)
        plt.ylabel("Energy Load (GW)", fontsize=LABEL_SIZE)
        plt.xticks(fontsize=TICK_SIZE)
        plt.yticks(fontsize=TICK_SIZE)
        plt.title("Dependence of Energy Load on Temperature", fontsize=TITLE_SIZE)
        plt.legend(fontsize=LEGEND_SIZE)
        plt.grid(color='gray', linestyle="--", alpha=0.5)
        plt.tight_layout()

        # Save plot
        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)  # Create directory if it doesn't exist

            file_name = "scatter_temp_vs_load_with_trend.png"
            full_path = os.path.join(save_path, file_name)
            plt.savefig(full_path, dpi=300)
            print(f"Plot saved to {full_path}")

        # Show plot
        # plt.show()

    except Exception as e:
        print(f"Error while creating scatter plot: {e}")


def plot_bridgeday_patterns(df_data, df_holidays, y_column, y_label, title_base, save_path=None):
    """
    Creates plots comparing patterns between regular days, holidays, and bridge days.
    """
    try:
        # KIT Colors
        KIT_COLORS = {
            'gruen': (0 / 255, 150 / 255, 130 / 255),  # Regular days
            'blau': (70 / 255, 100 / 255, 170 / 255),  # Bridge days
            'orange': (223 / 255, 155 / 255, 27 / 255)  # Holidays
        }

        TITLE_SIZE = 20
        LABEL_SIZE = 20
        TICK_SIZE = 20
        LEGEND_SIZE = 20

        # Create a copy and ensure datetime
        df_analysis = df_data.copy()
        df_analysis['dateTime'] = pd.to_datetime(df_analysis['dateTime'])

        # Convert holiday dates to datetime and create holiday flag
        if 'Date' in df_holidays.columns:  # Karlsruhe holidays
            df_holidays['Date'] = pd.to_datetime(df_holidays['Date'], format='%d.%m.%y')
            holiday_dates = set(df_holidays['Date'].dt.strftime('%Y-%m-%d'))
        else:  # German holidays
            holiday_dates = set(pd.to_datetime(df_holidays['Datum']).dt.strftime('%Y-%m-%d'))

        # Create initial flags
        df_analysis['date_str'] = df_analysis['dateTime'].dt.strftime('%Y-%m-%d')
        df_analysis['is_holiday'] = df_analysis['date_str'].isin(holiday_dates)
        df_analysis['weekday'] = df_analysis['dateTime'].dt.dayofweek
        df_analysis['hour'] = df_analysis['dateTime'].dt.hour
        df_analysis['month'] = df_analysis['dateTime'].dt.month

        # Identify bridge days
        df_analysis['bridge_day'] = False
        date_series = pd.to_datetime(df_analysis['date_str'].unique())

        for date in date_series:
            if date.weekday() < 5:  # Weekday
                prev_day = (date - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                next_day = (date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

                is_prev_holiday = prev_day in holiday_dates
                is_next_holiday = next_day in holiday_dates

                if (is_prev_holiday or is_next_holiday):
                    df_analysis.loc[df_analysis['date_str'] == date.strftime('%Y-%m-%d'), 'bridge_day'] = True

        # 1. Daily Pattern Comparison
        plt.figure(figsize=(15, 6))

        regular_hourly = df_analysis[~(df_analysis['is_holiday'] | df_analysis['bridge_day'])].groupby('hour')[
            y_column].mean()
        holiday_hourly = df_analysis[df_analysis['is_holiday']].groupby('hour')[y_column].mean()
        bridge_hourly = df_analysis[df_analysis['bridge_day']].groupby('hour')[y_column].mean()

        plt.plot(regular_hourly.index, regular_hourly.values,
                 label='Regular Days', color=KIT_COLORS['gruen'], linewidth=2)
        plt.plot(holiday_hourly.index, holiday_hourly.values,
                 label='Holidays', color=KIT_COLORS['orange'], linewidth=2)
        plt.plot(bridge_hourly.index, bridge_hourly.values,
                 label='Bridge Days', color=KIT_COLORS['blau'], linewidth=2)

        plt.xlabel('Hour of Day', fontsize=LABEL_SIZE)
        plt.ylabel(y_label, fontsize=LABEL_SIZE)
        plt.title(f'{title_base}: Daily Patterns Comparison', fontsize=TITLE_SIZE)
        plt.xticks(fontsize=TICK_SIZE)
        plt.yticks(fontsize=TICK_SIZE)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=LEGEND_SIZE)

        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(os.path.join(save_path, f"{title_base.replace(' ', '_').lower()}_daily_patterns.png"),
                        dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Monthly Distribution
        plt.figure(figsize=(15, 6))

        regular_monthly = df_analysis[~(df_analysis['is_holiday'] | df_analysis['bridge_day'])].groupby('month')[
            y_column].mean()
        holiday_monthly = df_analysis[df_analysis['is_holiday']].groupby('month')[y_column].mean()
        bridge_monthly = df_analysis[df_analysis['bridge_day']].groupby('month')[y_column].mean()

        plt.plot(range(1, 13), [regular_monthly.get(m, float('nan')) for m in range(1, 13)],
                 label='Regular Days', color=KIT_COLORS['gruen'], marker='o', linewidth=2)
        plt.plot(range(1, 13), [holiday_monthly.get(m, float('nan')) for m in range(1, 13)],
                 label='Holidays', color=KIT_COLORS['orange'], marker='o', linewidth=2)
        plt.plot(range(1, 13), [bridge_monthly.get(m, float('nan')) for m in range(1, 13)],
                 label='Bridge Days', color=KIT_COLORS['blau'], marker='o', linewidth=2)

        plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize=TICK_SIZE)
        plt.xlabel('Month', fontsize=LABEL_SIZE)
        plt.ylabel(y_label, fontsize=LABEL_SIZE)
        plt.title(f'{title_base}: Monthly Patterns Comparison', fontsize=TITLE_SIZE)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=LEGEND_SIZE)

        if save_path:
            plt.savefig(os.path.join(save_path, f"{title_base.replace(' ', '_').lower()}_monthly_patterns.png"),
                        dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Box Plot
        plt.figure(figsize=(15, 6))

        data_to_plot = [
            df_analysis[~(df_analysis['is_holiday'] | df_analysis['bridge_day'])][y_column],
            df_analysis[df_analysis['is_holiday']][y_column],
            df_analysis[df_analysis['bridge_day']][y_column]
        ]

        plt.boxplot(data_to_plot, labels=['Regular Days', 'Holidays', 'Bridge Days'])
        plt.ylabel(y_label, fontsize=LABEL_SIZE)
        plt.xticks(fontsize=TICK_SIZE)
        plt.yticks(fontsize=TICK_SIZE)
        plt.title(f'{title_base}: Distribution Comparison', fontsize=TITLE_SIZE)
        plt.grid(True, linestyle='--', alpha=0.7)

        if save_path:
            plt.savefig(os.path.join(save_path, f"{title_base.replace(' ', '_').lower()}_distribution.png"),
                        dpi=300, bbox_inches='tight')
        plt.close()

        # Print statistics
        print(f"\nStatistics for {title_base}:")
        regular_mean = df_analysis[~(df_analysis['is_holiday'] | df_analysis['bridge_day'])][y_column].mean()
        holiday_mean = df_analysis[df_analysis['is_holiday']][y_column].mean()
        bridge_mean = df_analysis[df_analysis['bridge_day']][y_column].mean()

        print(f"Average on regular days: {regular_mean:.2f}")
        print(f"Average on holidays: {holiday_mean:.2f} ({((holiday_mean / regular_mean) - 1) * 100:.1f}% vs regular)")
        print(f"Average on bridge days: {bridge_mean:.2f} ({((bridge_mean / regular_mean) - 1) * 100:.1f}% vs regular)")

    except Exception as e:
        print(f"Error while creating plots: {e}")
        raise

df_germanHolidays, df_karlsruheHolidays = importHolidays()

df_energy = cleanEnergy(importEnergy())
save_path_energy = "/Users/luisafaust/Desktop/PTFSC_Data/Graphs/Bericht_Fin/Energy"
# plot_weekly_energy_lines(df_energy, start_date='2023-01-01', end_date='2024-01-01', save_path=save_path_energy)


df_bike_hourly =importBikehourly()
df_bike_hourly = preprocessBikeHourly(df_bike_hourly)
df_bike_hourly['bike_count'] = df_bike_hourly['bike_count'].astype(int)

save_path_bike = "/Users/luisafaust/Desktop/PTFSC_Data/Graphs/Bericht_Fin/Bike"
# plot_weekly_bike_lines(df_bike_hourly, start_date='2023-01-01', end_date='2024-01-01', save_path=save_path_bike)


# Energy
weather_data_zwischenspeicher_energy = '/Users/luisafaust/Desktop/PTFSC_Data/Weather/Energy_Zwischenspeicher/historic_recent_forecast_zwischenspeicher_2024-12-19.csv'
df_weather_energy = pd.read_csv(weather_data_zwischenspeicher_energy)
df_weather_energy['dateTime'] = pd.to_datetime(df_weather_energy['date'])
df_weather_energy['hour'] = df_weather_energy['hour'].astype(int)

# Bike
weather_data_zwischenspeicher_bike = '/Users/luisafaust/Desktop/PTFSC_Data/Weather/Bike_Zwischenspeicher/historic_recent_forecast_zwischenspeicher_2024-12-19.csv'

df_weather_bike = pd.read_csv(weather_data_zwischenspeicher_bike)
df_weather_bike = df_weather_bike.drop(df_weather_bike.columns[0], axis=1)
df_weather_bike['date'] = pd.to_datetime(df_weather_bike['date'])
df_weather_bike['hour'] = df_weather_bike['hour'].astype(int)


save_path_weather = '/Users/luisafaust/Desktop/PTFSC_Data/Graphs/Bericht_Fin'




plot_seasonal_patterns_kit(
    df=df_energy,
    y_column='load',
    y_label='Energy Load (GW)',
    title_base='Energy Load',
    save_path=save_path_energy
)

plot_seasonal_patterns_kit(
    df=df_bike_hourly,
    y_column='bike_count',
    y_label='Bike Count',
    title_base='Bike Count',
    save_path=save_path_bike
)

scatter_with_trend_bike_daily(
    df_weather=df_weather_bike,
    df_bike_hourly=df_bike_hourly,
    start_date='2023-01-01',
    end_date='2024-01-01',
    color='green',
    save_path=save_path_weather
)

scatter_with_trend_energy(
    df_weather=df_weather_energy,
    df_energy=df_energy,
    start_date='2023-01-01',
    end_date='2024-01-01',
    color='green',
    save_path=save_path_weather
)


plot_bridgeday_patterns(
    df_data=df_energy,
    df_holidays=df_germanHolidays,
    y_column='load',
    y_label='Energy Load (GW)',
    title_base='Energy Load',
    save_path=save_path_energy
)

plot_bridgeday_patterns(
    df_data=df_bike_hourly,
    df_holidays=df_karlsruheHolidays,
    y_column='bike_count',
    y_label='Bike Count',
    title_base='Bike Count',
    save_path=save_path_bike
)