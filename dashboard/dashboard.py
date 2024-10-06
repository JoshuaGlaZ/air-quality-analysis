import calendar
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

sns.set(style='dark')


def create_aqi_df(data_df):
    breakpoints = {
        'PM2.5': [(0, 12, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150), (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300)],
        'PM10': [(0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150), (255, 354, 151, 200), (355, 424, 201, 300)],
        'SO2': [(0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150), (186, 304, 151, 200), (305, 604, 201, 300)],
        'NO2': [(0, 53, 0, 50), (54, 100, 51, 100), (101, 360, 101, 150), (361, 649, 151, 200), (650, 1249, 201, 300)],
        'O3': [(0, 50, 0, 50), (51, 100, 51, 100), (101, 168, 101, 150), (169, 208, 151, 200), (209, 748, 201, 300)]
    }

    def truncate_concentration(value, pollutant):
        if pollutant == 'PM2.5':
            return round(value, 1)
        elif pollutant == 'PM10':
            return int(value)
        elif pollutant == 'SO2' or pollutant == 'NO2':
            return int(value)
        elif pollutant == 'CO':
            print(round(value, 2))
            return round(value, 2)
        elif pollutant == 'O3':
            return round(value, 3)
        else:
            return value

    def calculate_aqi(value, breakpoints, pollutant):
        value = truncate_concentration(value, pollutant)

        for C_Lo, C_Hi, I_Lo, I_Hi in breakpoints:
            if C_Lo <= value <= C_Hi:
                I = (I_Hi - I_Lo) / (C_Hi - C_Lo) * (value - C_Lo) + I_Lo
                return round(I)
        return 0

    def calculate_overall_aqi(row, pollutants):
        aqi_values = []
        for pollutant in pollutants:
            if pollutant in row:
                concentration = row[pollutant]
                aqi = calculate_aqi(
                    concentration, breakpoints[pollutant], pollutant)
                aqi_values.append(aqi)
        return max(aqi_values) if aqi_values else 0  # Return the maximum AQI

    df_districts = data_df.groupby(['station', 'year', 'month']).agg({
        'PM2.5': 'mean',
        'PM10': 'mean',
        'NO2': 'mean',
        'SO2': 'mean',
        'O3': 'mean'
    }).reset_index()
    pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'O3']

    df_districts['AQI'] = df_districts.apply(
        lambda row: calculate_overall_aqi(row, pollutants), axis=1)

    df_districts_grouped = df_districts.groupby(
        'station').agg({'AQI': 'mean'}).reset_index()
    df_districts_grouped['AQI'] = df_districts_grouped['AQI'].apply(
        lambda x: "{:,.1f}".format(x)).astype(float)
    df_districts_grouped = df_districts_grouped.sort_values(
        by='AQI', ascending=False)

    return df_districts_grouped


def create_safelevel_df(data_df):
    df_safelevel = data_df.groupby(['station', 'year', 'month', 'day', 'hour']).agg({
        'PM2.5': 'mean',
        'PM10': 'mean',
        'NO2': 'mean',
        'SO2': 'mean',
        'O3': 'mean'
    }).reset_index()

    df_safelevel['datetime'] = pd.to_datetime(
        df_safelevel[['year', 'month', 'day', 'hour']])
    df_safelevel.set_index('datetime', inplace=True)
    df_safelevel.sort_index(inplace=True)

    df_safelevel['PM2.5_mean'] = df_safelevel['PM2.5'].rolling(window=24).mean()
    df_safelevel['PM10_mean'] = df_safelevel['PM10'].rolling(window=24).mean()
    df_safelevel['SO2_mean'] = df_safelevel['SO2'].rolling(window=24).mean()
    df_safelevel['NO2_mean'] = df_safelevel['NO2'].rolling(window=24).mean()
    df_safelevel['O3_mean'] = df_safelevel['O3'].rolling(window=8).mean()
    return df_safelevel


def create_air_quality_hourly_df(data_df):
    df_safelevel = create_safelevel_df(data_df)
    avg_pollutants = df_safelevel.groupby(['hour'])[
        ['PM2.5_mean', 'PM10_mean', 'SO2_mean', 'NO2_mean', 'O3_mean']].mean().reset_index()
    avg_pollutants_melted = avg_pollutants.melt(id_vars=['hour'],
                                                var_name='Pollutant',
                                                value_name='Average Level')
    return avg_pollutants_melted


def create_air_quality_monthly_df(data_df):
    df_safelevel = create_safelevel_df(data_df)
    avg_pollutants_monthly = df_safelevel.groupby('month')[
        ['PM2.5_mean', 'PM10_mean', 'SO2_mean', 'NO2_mean', 'O3_mean']].mean().reset_index()
    avg_pollutants_monthly_melted = avg_pollutants_monthly.melt(id_vars=['month'],
                                                                var_name='Pollutant',
                                                                value_name='Average Level')
    return avg_pollutants_monthly_melted


def create_pollutant_correlation_df(data_df):
    return data_df[['PM2.5', 'PM10', 'SO2', 'NO2', 'O3', 'TEMP (C)']].corr()
  

def create_hexbins_temp_df(data_df):
    return data_df[['PM2.5', 'PM10', 'SO2', 'NO2', 'O3', 'TEMP (C)']]


root_dir = Path(__file__).parent  
path = Path(__file__).parent /  "PRSA_Data_stations.csv"  
station_df = pd.read_csv(path)
station_df['datetime'] = pd.to_datetime(station_df[['year', 'month', 'day', 'hour']])
station_df.set_index('datetime', inplace=True)
station_df.reset_index(inplace=True)

min_date = station_df["datetime"].min()
max_date = station_df["datetime"].max()

with st.sidebar:
    st.image("https://openweather.co.uk/storage/app/uploads/public/227/_Ke/ep%20/227_Keep%20your%20finger%20on%20the%20pulse%20of%20Air%20Pollution%20API.jpg")

    start_date, end_date = st.date_input(
        label='Rentang Waktu', min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

main_df = station_df[(station_df["datetime"] >= str(start_date)) &
                     (station_df["datetime"] <= str(end_date))]

aqi_df = create_aqi_df(main_df)
air_quality_hourly_df = create_air_quality_hourly_df(main_df)
air_quality_monthly_df = create_air_quality_monthly_df(main_df)


def aqi_color(aqi):
    if aqi <= 50:
        return 'green'  # Good
    elif aqi <= 100:
        return 'yellow'  # Moderate
    elif aqi <= 150:
        return 'orange'  # Unhealthy for Sensitive Groups
    elif aqi <= 200:
        return 'red'  # Unhealthy
    elif aqi <= 300:
        return 'purple'  # Very Unhealthy
    else:
        return 'brown'  # Hazardous


def aqi_level(aqi):
    if aqi <= 50:
        return 'Good'
    elif aqi <= 100:
        return 'Moderate'
    elif aqi <= 150:
        return 'Unhealthy for Sensitive Groups'
    elif aqi <= 200:
        return 'Unhealthy'
    elif aqi <= 300:
        return 'Very Unhealthy'
    else:
        return 'Hazardous'


st.header('Air Quality Dashboard :cloud:')

st.subheader('Highest AQI Level from Stations')
max_aqi_station = aqi_df.loc[aqi_df['AQI'].idxmax()]

st.metric(label="Station", value=max_aqi_station['station'])
col1, col2 = st.columns(2)

with col1:
    st.metric(label="AQI Level", value=f"{max_aqi_station['AQI']:.1f}")

with col2:
    aqi_status = aqi_level(max_aqi_station['AQI'])
    st.metric(label="AQI Category", value=aqi_status)


sns.set_context("talk")
sns.set_style("whitegrid")

plt.figure(figsize=(18, 12))

barplot = sns.barplot(data=aqi_df, x='station', y='AQI',
                      palette=[aqi_color(aqi) for aqi in aqi_df['AQI']])

plt.xticks(ha='center', va='center', fontsize=14)
plt.xlabel('Station', fontsize=20)
plt.ylabel('AQI Level', fontsize=20)

legend_labels = ['Good (0-50)', 'Moderate (51-100)', 'Unhealthy for Sensitive Groups (101-150)',
                 'Unhealthy (151-200)', 'Very Unhealthy (201-300)', 'Hazardous (301-500)']
legend_colors = ['green', 'yellow', 'orange', 'red', 'purple', 'brown']

handles = [plt.Rectangle((0, 0), 1, 1, color=legend_colors[i])
           for i in range(len(legend_labels))]
plt.legend(handles, legend_labels,
           title="Air Pollution Level", loc='lower right')

for bar in barplot.patches:
    bar_height = bar.get_height()
    barplot.annotate(f'{bar_height:.1f}',
                     (bar.get_x() + bar.get_width() / 2, bar_height),
                     ha='center', va='bottom', fontsize=14,
                     color='black' if bar_height < 200 else 'white')

plt.grid(axis='y', linestyle='--', alpha=0.7)
sns.despine()

plt.tight_layout()
st.pyplot(plt)

who_limits = {
    'PM2.5_mean': 15,
    'PM10_mean': 45,
    'SO2_mean': 40,
    'NO2_mean': 25,
    'O3_mean': 100
}

palette = {
    'PM2.5_mean': 'darkorange',
    'PM10_mean': 'purple',
    'SO2_mean': 'green',
    'NO2_mean': 'cyan',
    'O3_mean': 'magenta'
}

legend_labels = {
    'PM2.5_mean': 'PM2.5 (µg/m³)',
    'PM10_mean': 'PM10 (µg/m³)',
    'SO2_mean': 'Sulfur Dioxide (µg/m³)',
    'NO2_mean': 'Nitrogen Dioxide (µg/m³)',
    'O3_mean': 'Ozone (µg/m³)'
}

exceeding_pollutants = []
for pollutant in who_limits.keys():
    count_exceedances = (air_quality_hourly_df[air_quality_hourly_df['Pollutant'] == pollutant]['Average Level'] > who_limits[pollutant]).sum()
    if count_exceedances > 12:  # 50% of 24 hours
        exceeding_pollutants.append(pollutant)

st.subheader("Pollutants by Hours of Day")
if exceeding_pollutants:
    st.write('Exceeding WHO Limits: '+ ", ".join([legend_labels[pollutant].split('(')[0] for pollutant in exceeding_pollutants]))
else:
    st.write("No pollutants exceed the limits.")

col1, col2 = st.columns([2, 1])  

with col1:
    plt.figure(figsize=(18, 12))

    sns.lineplot(data=air_quality_hourly_df, 
                 x='hour', 
                 y='Average Level', 
                 hue='Pollutant', 
                 palette=palette, 
                 marker='o',
                 linewidth=2.5,
                 alpha=0.8)

    for pollutant, limit in who_limits.items():
        plt.axhline(y=limit, color=palette[pollutant], linestyle='--', 
                        linewidth=3, label=f'{pollutant.split("_")[0]} WHO Limit') 

    plt.xlabel('Hour of the Day', fontsize=20)
    plt.ylabel('Average Pollutant Level (µg/m³)', fontsize=20)
    plt.xticks(range(0, 24))
    plt.xlim(-1, 24)
    plt.legend('', frameon=False)

    plt.tight_layout()
    st.pyplot(plt)

with col2:
    legend_data = {
        'Pollutant': [f'<span style="display:inline-block;color:{palette[pollutant]}; font-weight:bold;">■</span> {pollutant.split("_")[0]}' for pollutant in palette.keys()],
        'WHO Limit (µg/m³)': [who_limits[pollutant] for pollutant in palette.keys()],
    }

    legend_df = pd.DataFrame(legend_data)
    st.markdown(legend_df.style.hide(axis="index").to_html(), unsafe_allow_html=True)
    

st.subheader("Pollutants by Months")

col1, col2 = st.columns([2, 1])  

with col1:
    plt.figure(figsize=(18, 12))
    sns.barplot(data=air_quality_monthly_df, 
                x='month', 
                y='Average Level', 
                hue='Pollutant', 
                palette=palette)

    unique_months = sorted(air_quality_monthly_df['month'].unique())
    month_labels = [calendar.month_abbr[month] for month in unique_months]

    plt.xlabel('Month', fontsize=20)
    plt.ylabel('Average Pollutant Level (µg/m³)', fontsize=20)
    plt.xticks(np.arange(len(unique_months)), labels=month_labels)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.legend('', frameon=False)

    plt.tight_layout()
    st.pyplot(plt)
    
with col2:
    legend_data = {
        'Pollutant': [f'<span style="display:inline-block;color:{palette[pollutant]}; font-weight:bold;">■</span> {pollutant.split("_")[0]}' for pollutant in palette.keys()],
    }

    legend_df = pd.DataFrame(legend_data)
    st.markdown(legend_df.style.hide(axis="index").to_html(), unsafe_allow_html=True)
    
st.subheader("Correlation Heatmap of Air Quality and Temperature  ")
    
corr_matrix = create_pollutant_correlation_df(main_df)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

corr_matrix_unstacked = corr_matrix.unstack()
filtered_corr = corr_matrix_unstacked[corr_matrix_unstacked != 1.0]

top_positive_corr = filtered_corr.sort_values(ascending=False).head(3)
top_negative_corr = filtered_corr.sort_values(ascending=True).head(3)

col1, col2 = st.columns([1, 2])

with col1:
    col11, col12 = st.columns(2)
    
    with col11:
        st.write("Top 3 Positive Correlations")
        for (var1, var2), value in top_positive_corr.items():
            st.metric(label=f"{var1} and {var2}", value=f"{value:.2f}")

    with col12:
        st.write("Top 3 Negative Correlations")
        for (var1, var2), value in top_negative_corr.items():
            st.metric(label=f"{var1} and {var2}", value=f"{value:.2f}")
    
with col2:    
    plt.figure(figsize=(16, 12))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='vlag', linewidths=0.5,
                square=True, cbar_kws={"shrink": .75}, vmin=-1, vmax=1)
    plt.xticks(rotation=45, ha='right', fontsize=20)
    plt.yticks(rotation=45, fontsize=20)

    plt.tight_layout()
    st.pyplot(plt)

tempsdf = create_hexbins_temp_df(main_df)
tabs = st.tabs(tempsdf.columns[:-1].tolist())

for i, variable in enumerate(tempsdf.columns[:-1]):
    with tabs[i]:
        st.subheader(f'Hexbin Plot of {variable} vs Temperature')
        
        if tempsdf[variable].corr(tempsdf['TEMP (C)']) > 0:
            st.write(f"  There appears to be a **positive** correlation between {variable} and temperature")
        elif tempsdf[variable].corr(tempsdf['TEMP (C)']) < 0:
            st.write(f"  There appears to be a **negative** correlation between {variable} and temperature")
        else:
            st.write(f"  No clear correlation is evident between {variable} and temperature")

        plt.figure(figsize=(10, 6))
        sns.histplot(x=tempsdf['TEMP (C)'], y=tempsdf[variable],
                     bins=30, pmax=0.8, cmap='Blues', cbar=True)

        plt.xlabel('Temperature (°C)', fontsize=20)
        plt.ylabel(f'{variable} Levels', fontsize=20)

        st.pyplot(plt)  
        plt.close() 
