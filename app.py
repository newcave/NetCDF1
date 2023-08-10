import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import folium
import netCDF4 as nc

# Create a Streamlit app
st.title('Rainfall Prediction - KMA')
st.sidebar.title('Upload NetCDF File')
uploaded_file = st.sidebar.file_uploader("Upload your NetCDF file", type=["nc"])

if uploaded_file is not None:
    # Load NetCDF data
    df = nc.Dataset(uploaded_file)
    df_var = df.variables['rain'][:]
    rain_array = np.array(df_var)

    # Load latitude and longitude data
    lat_data = pd.read_csv('./data/dongne_lat_info.txt', header=None).values
    lon_data = pd.read_csv('./data/dongne_lon_info.txt', header=None).values

    latitude_array = lat_data
    longitude_array = lon_data

    # Trim latitude and longitude data
    latitude_trimmed = latitude_array[:-1, :-1]
    longitude_trimmed = longitude_array[:-1, :-1]

    # Trim the rain_array to match the shape of trimmed latitude and longitude arrays
    rain_trimmed = rain_array[:latitude_trimmed.shape[0], :latitude_trimmed.shape[1]]

    # Display the heatmap
    aspect_ratio = (longitude_trimmed.max() - longitude_trimmed.min()) / (latitude_trimmed.max() - latitude_trimmed.min())
    fig = plt.figure(figsize=(6 * aspect_ratio, 6))
    plt.imshow(rain_trimmed, cmap='rainbow', extent=[longitude_trimmed.min(), longitude_trimmed.max(), latitude_trimmed.min(), latitude_trimmed.max()], vmax=5)
    plt.colorbar(label='mm/hr')
    plt.title('Rainfall_Pred_KMA')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    st.pyplot(fig)
else:
    st.write("Please upload a NetCDF file using the sidebar.")
