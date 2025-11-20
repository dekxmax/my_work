import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta

API_KEY = "95ce2dfa2a784f0b979104501252607"
HIST_API_URL = "http://api.weatherapi.com/v1/history.json"
FORECAST_API_URL = "http://api.weatherapi.com/v1/forecast.json"

# Safety tips dictionary
SAFETY_TIPS = {
    "extreme_temp": {
        "title": "Extreme Temperature Safety",
        "tips": [
            "Stay hydrated by drinking plenty of water",
            "Avoid strenuous outdoor activities during peak heat hours (10am-4pm)",
            "Wear lightweight, light-colored, loose-fitting clothing",
            "Never leave children or pets in parked vehicles",
            "Check on elderly neighbors and those without air conditioning"
        ]
    },
    "extreme_rain": {
        "title": "Heavy Rainfall Safety",
        "tips": [
            "Avoid walking or driving through flood waters",
            "Be aware of potential flash flooding in low-lying areas",
            "Ensure proper drainage around your property",
            "Have emergency supplies ready in case of evacuation",
            "Stay informed about local weather updates"
        ]
    },
    "extreme_wind": {
        "title": "High Wind Safety",
        "tips": [
            "Secure outdoor objects that could blow away",
            "Stay away from windows during high winds",
            "Avoid unnecessary travel",
            "Be alert for falling tree limbs or power lines",
            "Have a flashlight and batteries in case of power outages"
        ]
    }
}

def fetch_city_coords_and_country(city, api_key=API_KEY):
    params = {
        "key": api_key,
        "q": city,
        "days": 1,
        "aqi": "no",
        "alerts": "no"
    }
    resp = requests.get(FORECAST_API_URL, params=params)
    resp.raise_for_status()
    data = resp.json()
    loc = data['location']
    return loc['lat'], loc['lon'], loc['country'], loc.get('region', '')

@st.cache_data(show_spinner=False)
def fetch_historical_weather_weatherapi(city, start_date, end_date, api_key=API_KEY):
    lat, lon, country, region = fetch_city_coords_and_country(city, api_key)
    all_days = []
    date = start_date
    while date <= end_date:
        params = {
            "key": api_key,
            "q": city,
            "dt": date.strftime('%Y-%m-%d'),
        }
        resp = requests.get(HIST_API_URL, params=params)
        try:
            data = resp.json()
        except Exception:
            date += timedelta(days=1)
            continue
        if resp.status_code != 200 or 'forecast' not in data or 'forecastday' not in data['forecast']:
            date += timedelta(days=1)
            continue
        for day in data['forecast']['forecastday']:
            d = day['day']
            hour_data = []
            for hour in day['hour']:
                hour_data.append({
                    'time': pd.to_datetime(hour['time']),
                    'temp_c': hour['temp_c'],
                    'precip_mm': hour['precip_mm'],
                    'wind_kph': hour['wind_kph'],
                    'humidity': hour['humidity'],
                    'feelslike_c': hour['feelslike_c'],
                    'condition': hour['condition']['text']
                })
            all_days.append({
                'city': data['location']['name'],
                'state': region,
                'country': country,
                'date': pd.to_datetime(day['date']),
                'temp_day': d['avgtemp_c'],
                'temp_min': d['mintemp_c'],
                'temp_max': d['maxtemp_c'],
                'rain': d['totalprecip_mm'],
                'wind_speed': d['maxwind_kph'],
                'humidity': d['avghumidity'],
                'condition': d['condition']['text'],
                'lat': lat,
                'lon': lon,
                'hourly_data': hour_data
            })
        date += timedelta(days=1)
    if all_days:
        return pd.DataFrame(all_days)
    return pd.DataFrame()

def fetch_forecast_weatherapi(city, days=7, api_key=API_KEY):
    params = {
        "key": api_key,
        "q": city,
        "days": days,
        "aqi": "no",
        "alerts": "no"
    }
    resp = requests.get(FORECAST_API_URL, params=params)
    resp.raise_for_status()
    data = resp.json()
    loc = data['location']
    forecast_days = []
    for day in data['forecast']['forecastday']:
        d = day['day']
        forecast_days.append({
            'city': loc['name'],
            'state': loc.get('region', ''),
            'country': loc['country'],
            'date': pd.to_datetime(day['date']),
            'temp_day': d['avgtemp_c'],
            'temp_min': d['mintemp_c'],
            'temp_max': d['maxtemp_c'],
            'rain': d['totalprecip_mm'],
            'wind_speed': d['maxwind_kph'],
            'humidity': d['avghumidity'],
            'condition': d['condition']['text'],
            'lat': loc['lat'],
            'lon': loc['lon'],
        })
    return pd.DataFrame(forecast_days)

def detect_extremes(df, temp_thresh=35, rain_thresh=30, wind_thresh=50):
    df['extreme_temp'] = df['temp_max'] > temp_thresh
    df['extreme_rain'] = df['rain'] > rain_thresh
    df['extreme_wind'] = df['wind_speed'] > wind_thresh
    df['extreme_any'] = df[['extreme_temp', 'extreme_rain', 'extreme_wind']].any(axis=1)
    return df

def plot_temperature_trend(df, city, highlight_extremes=True):
    data = df[df['city'] == city].copy()
    data['date_str'] = data['date'].dt.strftime('%b %d')

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    ax.fill_between(data['date_str'], data['temp_min'], data['temp_max'],
                    color='skyblue', alpha=0.3, label='Daily Range')
    sns.lineplot(data=data, x='date_str', y='temp_day', marker='o',
                 color='royalblue', linewidth=2.5, label='Avg Temp', ax=ax)
    if highlight_extremes and data['extreme_temp'].any():
        extreme_data = data[data['extreme_temp']]
        ax.scatter(extreme_data['date_str'], extreme_data['temp_max'],
                   color='red', s=100, label='Extreme Heat', zorder=5)
    ax.set_title(f"Temperature Trend - {city}", fontsize=16, pad=20)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Temperature (°C)", fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig

def plot_hourly_weather(hourly_data, city, date):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Hourly Weather Details - {city} ({date.date()})", fontsize=16, y=1.02)
    sns.lineplot(data=hourly_data, x='time', y='temp_c', ax=axes[0, 0], marker='o', color='red')
    axes[0, 0].set_title("Temperature (°C)")
    axes[0, 0].set_xlabel("")
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)
    sns.barplot(data=hourly_data, x='time', y='precip_mm', ax=axes[0, 1], color='blue')
    axes[0, 1].set_title("Precipitation (mm)")
    axes[0, 1].set_xlabel("")
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)
    sns.lineplot(data=hourly_data, x='time', y='wind_kph', ax=axes[1, 0], marker='o', color='green')
    axes[1, 0].set_title("Wind Speed (kph)")
    axes[1, 0].set_xlabel("")
    axes[1, 0].grid(True, linestyle='--', alpha=0.5)
    sns.lineplot(data=hourly_data, x='time', y='humidity', ax=axes[1, 1], marker='o', color='purple')
    axes[1, 1].set_title("Humidity (%)")
    axes[1, 1].set_xlabel("")
    axes[1, 1].grid(True, linestyle='--', alpha=0.5)
    for ax in axes.flat:
        ax.set_xticklabels([pd.to_datetime(t.get_text()).strftime('%H:%M') for t in ax.get_xticklabels()],
                           rotation=45, ha='right')
    plt.tight_layout()
    return fig

def plot_comparison(df, metric):
    metric_labels = {
        'temp_day': 'Average Temperature (°C)',
        'temp_max': 'Maximum Temperature (°C)',
        'temp_min': 'Minimum Temperature (°C)',
        'rain': 'Precipitation (mm)',
        'wind_speed': 'Wind Speed (kph)',
        'humidity': 'Humidity (%)'
    }
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    palette = sns.color_palette("husl", len(df['city'].unique()))
    sns.lineplot(data=df, x='date', y=metric, hue='city',
                 marker='o', linewidth=2.5, palette=palette, ax=ax)
    ax.set_title(f"{metric_labels.get(metric, metric.replace('_', ' ').title())} Comparison",
                 fontsize=16, pad=20)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel(metric_labels.get(metric, metric.replace('_', ' ').title()), fontsize=12)
    ax.legend(title='City', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, linestyle='--', alpha=0.5)
    return fig

def show_map(df):
    city_groups = df.groupby(['city', 'state', 'country', 'lat', 'lon'])
    m = folium.Map(location=[df['lat'].mean(), df['lon'].mean()], zoom_start=2)
    for (city, state, country, lat, lon), group in city_groups:
        extreme_days = group['extreme_any'].sum()
        total_days = len(group)
        popup_html = f"""
        <div style="width: 250px;">
            <h4 style="margin:0;padding:0;color:#2b5876;">{city}, {state}</h4>
            <p style="margin:0;padding:0;color:#4e4376;">{country}</p>
            <hr style="margin:5px 0;border-color:#eee;">
            <p style="margin:3px 0;"><b>Period:</b> {group['date'].min().date()} to {group['date'].max().date()}</p>
            <p style="margin:3px 0;"><b>Extreme Days:</b> {extreme_days} of {total_days}</p>
            <p style="margin:3px 0;"><b>Max Temp:</b> {group['temp_max'].max():.1f}°C</p>
            <p style="margin:3px 0;"><b>Max Rain:</b> {group['rain'].max():.1f}mm</p>
            <p style="margin:3px 0;"><b>Max Wind:</b> {group['wind_speed'].max():.1f} kph</p>
        </div>
        """
        extreme_percent = extreme_days / total_days
        if extreme_percent > 0.5:
            color = 'red'
        elif extreme_percent > 0.2:
            color = 'orange'
        else:
            color = 'green'
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{city}, {state}",
            icon=folium.Icon(color=color, icon='info-sign')
        ).add_to(m)
    return m

def generate_city_report(df, city):
    city_data = df[df['city'] == city].copy()
    if city_data.empty:
        return None
    city_data['date_str'] = city_data['date'].dt.strftime('%b %d')
    latest_data = city_data.iloc[-1]
    report = {
        "overview": {
            "city": city,
            "state": latest_data['state'],
            "country": latest_data['country'],
            "period": f"{city_data['date'].min().date()} to {city_data['date'].max().date()}",
            "total_days": len(city_data),
            "extreme_days": int(city_data['extreme_any'].sum())
        },
        "stats": {
            "avg_temp": city_data['temp_day'].mean(),
            "max_temp": city_data['temp_max'].max(),
            "min_temp": city_data['temp_min'].min(),
            "total_rain": city_data['rain'].sum(),
            "max_rain": city_data['rain'].max(),
            "avg_wind": city_data['wind_speed'].mean(),
            "max_wind": city_data['wind_speed'].max(),
            "common_condition": city_data['condition'].mode()[0]
        },
        "extreme_events": city_data[city_data['extreme_any']].to_dict('records')
    }
    return report

def display_city_report(report):
    if not report:
        st.warning("No data available for this city.")
        return
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📍 City Overview")
        st.markdown(f"""
        **City:** {report['overview']['city']}  
        **State/Region:** {report['overview']['state']}  
        **Country:** {report['overview']['country']}  
        **Period:** {report['overview']['period']}  
        **Total Days:** {report['overview']['total_days']}  
        **Extreme Weather Days:** {report['overview']['extreme_days']}
        """)
        st.subheader("📊 Weather Statistics")
        st.markdown(f"""
        **Average Temperature:** {report['stats']['avg_temp']:.1f}°C  
        **Highest Temperature:** {report['stats']['max_temp']:.1f}°C  
        **Lowest Temperature:** {report['stats']['min_temp']:.1f}°C  
        **Total Precipitation:** {report['stats']['total_rain']:.1f} mm  
        **Max Daily Rain:** {report['stats']['max_rain']:.1f} mm  
        **Average Wind Speed:** {report['stats']['avg_wind']:.1f} kph  
        **Max Wind Speed:** {report['stats']['max_wind']:.1f} kph  
        **Most Common Condition:** {report['stats']['common_condition']}
        """)
    with col2:
        st.subheader("⚠ Extreme Weather Events")
        if report['extreme_events']:
            for event in report['extreme_events']:
                extremes = []
                if event['extreme_temp']:
                    extremes.append(f"🌡️ High Temp: {event['temp_max']}°C")
                if event['extreme_rain']:
                    extremes.append(f"🌧️ Heavy Rain: {event['rain']}mm")
                if event['extreme_wind']:
                    extremes.append(f"💨 Strong Wind: {event['wind_speed']}kph")
                st.markdown(f"""
                **{event['date'].strftime('%b %d, %Y')}**  
                {", ".join(extremes)}  
                Condition: {event['condition']}
                """)
                st.markdown("---")
        else:
            st.info("No extreme weather events detected for this period.")
    if report['extreme_events']:
        st.subheader("🛡️ Safety Precautions")
        extreme_types = set()
        for event in report['extreme_events']:
            if event['extreme_temp']:
                extreme_types.add('extreme_temp')
            if event['extreme_rain']:
                extreme_types.add('extreme_rain')
            if event['extreme_wind']:
                extreme_types.add('extreme_wind')
        for ext_type in extreme_types:
            tips = SAFETY_TIPS.get(ext_type, {})
            with st.expander(f"⚠ {tips.get('title', 'Safety Tips')}"):
                for tip in tips.get('tips', []):
                    st.markdown(f"- {tip}")

def main():
    st.set_page_config(page_title="Advanced Weather Dashboard", page_icon="🌦️", layout="wide")
    st.markdown("""
    <style>
    .css-1aumxhk, .css-1v3fvcr, .st-bb, .st-at, .st-ae, .css-1q1n0ol, .css-1q8dd3e, .css-1q8dd3e p, .css-1q8dd3e h1, .css-1q8dd3e h2, .css-1q8dd3e h3, .css-1q8dd3e h4, .css-1q8dd3e h5, .css-1q8dd3e h6, .markdown-text-container, .stMarkdown, .stCaption, .stAlert, .stDataFrame, .stMetric, .stTitle, .stHeader, .stSubheader {
        color: #ffffff !important;
    }
    .main {background-color: #101010;}
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #000000;
        color: #ffffff;
        border-radius: 4px 4px 0 0;
        padding: 10px 20px;
        margin-right: 5px;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #333333;
        color: #ffffff;
    }
    .stTabs [aria-selected="true"] {
        background-color: #000000;
        color: #ffffff;
    }
    .stAlert {
        padding: 20px;
        border-radius: 4px;
        color: #ffffff !important;
        background-color: #3a3a3a !important;
    }
    .stDataFrame {
        border: 1px solid #e0e0e0;
        color: #ffffff !important;
        background-color: #1a1a1a !important;
    }
    .css-1v3fvcr {
        padding: 1rem 1rem 0rem;
        background-color: #222222;
        color: #ffffff !important;
    }
    .stButton>button {
        background-color: #000000;
        color: #ffffff;
        border-radius: 4px;
    }
    .stButton>button:hover {
        background-color: #333333;
        color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)
    st.title("🌦️ Advanced Weather Analytics Dashboard")
    st.write("""
    Explore historical weather patterns, detect extreme events, and get safety recommendations.
    Data provided by WeatherAPI.com.
    """)
    with st.sidebar:
        st.header("🔍 Search Parameters")
        cities = st.text_input("Cities (comma-separated)", "London,Paris,New York,Tokyo")
        city_list = [city.strip() for city in cities.split(",") if city.strip()]
        today = datetime.today()
        min_date = today - timedelta(days=365)
        max_date = today
        start_date = st.date_input("Start Date", min_value=min_date, max_value=max_date,
                                   value=today - timedelta(days=30))
        end_date = st.date_input("End Date", min_value=min_date, max_value=max_date,
                                 value=today)
        if start_date > end_date:
            st.error("Start date must be before end date.")
            st.stop()
        with st.expander("⚙️ Extreme Weather Thresholds", expanded=False):
            temp_thresh = st.slider("Extreme Temperature (°C)", 30, 45, 35)
            rain_thresh = st.slider("Extreme Rainfall (mm)", 10, 100, 30)
            wind_thresh = st.slider("Extreme Wind Speed (kph)", 20, 100, 50)
        st.markdown("---")
        st.markdown("""
        **Note:**  
        - Data availability depends on WeatherAPI.com  
        - Historical data may be limited for some locations  
        - Processing may take a moment for longer date ranges
        """)
    if 'weather_data' not in st.session_state:
        st.session_state['weather_data'] = None
        st.session_state['last_params'] = {}
    params_now = {
        'cities': tuple(city_list),
        'start_date': start_date,
        'end_date': end_date,
        'temp_thresh': temp_thresh,
        'rain_thresh': rain_thresh,
        'wind_thresh': wind_thresh
    }
    fetch_clicked = st.sidebar.button("Fetch and Analyze Data", use_container_width=True)
    should_fetch = (
            fetch_clicked or
            st.session_state['weather_data'] is None or
            st.session_state['last_params'] != params_now
    )
    if should_fetch and city_list:
        all_dfs = []
        with st.spinner("Fetching historical weather data... (this may take a minute)"):
            progress_bar = st.progress(0)
            for i, city in enumerate(city_list):
                try:
                    df = fetch_historical_weather_weatherapi(city, start_date, end_date)
                    if not df.empty:
                        all_dfs.append(df)
                    progress_bar.progress((i + 1) / len(city_list))
                except Exception as e:
                    st.warning(f"Failed for {city}: {str(e)}")
                    progress_bar.progress((i + 1) / len(city_list))
        if all_dfs:
            df = pd.concat(all_dfs, ignore_index=True)
            df = detect_extremes(df, temp_thresh, rain_thresh, wind_thresh)
            st.session_state['weather_data'] = df
            st.session_state['last_params'] = params_now
            st.success("Data loaded successfully!")
        else:
            st.session_state['weather_data'] = None
            st.session_state['last_params'] = params_now
            st.error("No data could be fetched. Please check your inputs.")

    df = st.session_state['weather_data']

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["🌍 Global View", "📊 City Comparison", "📍 City Reports", "📥 Download Data", "🔮 Forecast"]
    )

    if df is not None and not df.empty:
        with tab1:
            st.subheader("Global Weather Overview")
            st.write("""
            The map below shows all analyzed locations with markers indicating extreme weather frequency.
            Red markers indicate more frequent extreme weather, while green indicates less frequent.
            """)
            m = show_map(df)
            st_folium(m, width=1200, height=500)
            st.markdown("---")
            st.subheader("Extreme Weather Events Summary")
            extreme_summary = df.groupby('city').agg({
                'extreme_temp': 'sum',
                'extreme_rain': 'sum',
                'extreme_wind': 'sum',
                'extreme_any': 'sum'
            }).reset_index()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Cities Analyzed", len(df['city'].unique()))
            with col2:
                st.metric("Total Extreme Days", df['extreme_any'].sum())
            with col3:
                st.metric("Most Extreme City",
                          extreme_summary.loc[extreme_summary['extreme_any'].idxmax(), 'city'],
                          f"{extreme_summary['extreme_any'].max()} days")
            st.dataframe(extreme_summary.sort_values('extreme_any', ascending=False),
                         hide_index=True, use_container_width=True)
        with tab2:
            st.subheader("Weather Metrics Comparison")
            st.write("Compare weather patterns across different cities.")
            metric = st.selectbox("Select metric to compare",
                                  ['temp_day', 'temp_max', 'temp_min', 'rain', 'wind_speed', 'humidity'],
                                  key='compare_metric')
            st.pyplot(plot_comparison(df, metric))
            st.markdown("---")
            st.subheader("Extreme Events Timeline")
            extreme_days = df[df['extreme_any']].copy()
            if not extreme_days.empty:
                extreme_days['date_str'] = extreme_days['date'].dt.strftime('%b %d, %Y')
                for _, row in extreme_days.iterrows():
                    with st.container(border=True):
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.subheader(row['city'])
                            st.caption(f"{row['state']}, {row['country']}")
                            st.caption(row['date_str'])
                        with col2:
                            extremes = []
                            if row['extreme_temp']:
                                extremes.append(f"🌡️ Extreme Heat: {row['temp_max']}°C")
                            if row['extreme_rain']:
                                extremes.append(f"🌧️ Heavy Rain: {row['rain']}mm")
                            if row['extreme_wind']:
                                extremes.append(f"💨 Strong Wind: {row['wind_speed']}kph")
                            st.markdown("**Extreme Conditions:** " + " | ".join(extremes))
                            st.markdown(f"**Weather Condition:** {row['condition']}")
            else:
                st.info("No extreme weather events detected for these thresholds.")
        with tab3:
            st.subheader("Detailed City Weather Reports")
            selected_city = st.selectbox("Select a city", df['city'].unique(), key='city_report')
            city_data = df[df['city'] == selected_city]
            report = generate_city_report(df, selected_city)
            display_city_report(report)
            st.markdown("---")
            st.subheader("Daily Temperature Trend")
            st.pyplot(plot_temperature_trend(df, selected_city))
            if not city_data.empty:
                selected_date = st.selectbox("Select date for hourly data",
                                             city_data['date'].dt.strftime('%b %d, %Y').unique(),
                                             key='hourly_date')
                selected_day_data = city_data[city_data['date'].dt.strftime('%b %d, %Y') == selected_date]
                if not selected_day_data.empty and selected_day_data.iloc[0]['hourly_data']:
                    st.pyplot(plot_hourly_weather(
                        pd.DataFrame(selected_day_data.iloc[0]['hourly_data']),
                        selected_city,
                        selected_day_data.iloc[0]['date']
                    ))
        with tab4:
            st.subheader("Download Weather Data")
            st.write("Export the collected weather data for further analysis.")
            csv = df.drop(columns=['hourly_data']).to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name="weather_data.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.markdown("---")
            st.subheader("Data Preview")
            st.dataframe(df.drop(columns=['hourly_data']).head(20), use_container_width=True)
        with tab5:
            st.subheader("Upcoming Weather Forecast")
            forecast_days = st.slider("Forecast Days", 1, 14, 7)
            forecast_dfs = []
            with st.spinner("Fetching forecast data..."):
                for city in city_list:
                    try:
                        forecast_df = fetch_forecast_weatherapi(city, days=forecast_days)
                        if not forecast_df.empty:
                            forecast_dfs.append(forecast_df)
                    except Exception as e:
                        st.warning(f"Failed for {city}: {e}")
            if forecast_dfs:
                forecast_df = pd.concat(forecast_dfs, ignore_index=True)
                forecast_df = detect_extremes(forecast_df, temp_thresh, rain_thresh, wind_thresh)
                st.write("## Forecast Table")
                st.dataframe(forecast_df, use_container_width=True)
                st.write("## Forecast Temperature Trend")
                selected_city = st.selectbox("Select city for forecast plot", forecast_df["city"].unique(), key="forecast_city")
                city_forecast = forecast_df[forecast_df["city"] == selected_city]
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(city_forecast["date"], city_forecast["temp_day"], marker='o', label="Avg Temp")
                ax.fill_between(city_forecast["date"], city_forecast["temp_min"], city_forecast["temp_max"], alpha=0.3, label="Min/Max Range")
                ax.set_title(f"Forecast Temperature Trend - {selected_city}")
                ax.set_xlabel("Date")
                ax.set_ylabel("Temperature (°C)")
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.7)
                plt.xticks(rotation=45)
                st.pyplot(fig)
                # Extreme events in forecast
                st.write("## Upcoming Extreme Events")
                extreme_events = forecast_df[forecast_df["extreme_any"]]
                if not extreme_events.empty:
                    for _, event in extreme_events.iterrows():
                        extremes = []
                        if event['extreme_temp']:
                            extremes.append(f"🌡️ High Temp: {event['temp_max']}°C")
                        if event['extreme_rain']:
                            extremes.append(f"🌧️ Heavy Rain: {event['rain']}mm")
                        if event['extreme_wind']:
                            extremes.append(f"💨 Strong Wind: {event['wind_speed']}kph")
                        st.markdown(f"""
                        **{event['city']} ({event['date'].strftime('%b %d, %Y')})**  
                        {", ".join(extremes)}  
                        Condition: {event['condition']}
                        """)
                        st.markdown("---")
                    # Safety tips for forecast extremes
                    extreme_types = set()
                    for _, event in extreme_events.iterrows():
                        if event['extreme_temp']:
                            extreme_types.add('extreme_temp')
                        if event['extreme_rain']:
                            extreme_types.add('extreme_rain')
                        if event['extreme_wind']:
                            extreme_types.add('extreme_wind')
                    for ext_type in extreme_types:
                        tips = SAFETY_TIPS.get(ext_type, {})
                        with st.expander(f"⚠ {tips.get('title', 'Safety Tips')}"):
                            for tip in tips.get('tips', []):
                                st.markdown(f"- {tip}")
                else:
                    st.info("No extreme weather events predicted for upcoming days.")
            else:
                st.info("No forecast data fetched.")
    elif df is not None and df.empty:
        st.warning("The fetched data is empty. Please try different parameters.")
    st.markdown("---")
    st.caption("""
    *This dashboard uses weather data from [WeatherAPI.com](https://www.weatherapi.com/).  
    For emergency weather alerts, always check your local meteorological service.*
    """)

if __name__ == "__main__":
    main()