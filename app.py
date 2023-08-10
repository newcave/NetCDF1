import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from folium import plugins
import netCDF4 as nc
import os

# Create a Streamlit app
st.title('Rainfall Prediction - KMA')
st.sidebar.title('Upload NetCDF File')
uploaded_file = st.sidebar.file_uploader("Upload your NetCDF file", type=["nc"])

show_map = st.sidebar.checkbox("Show Map")

if uploaded_file is not None:
    # Load NetCDF data
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.read())
    
    try:
        df = nc.Dataset(uploaded_file.name)
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

        if show_map:
            # Create a Folium map
            m = folium.Map(location=[latitude_trimmed.mean(), longitude_trimmed.mean()], zoom_start=10)

            # Add heatmap layer to the map using Folium's HeatMap class
            heatmap_data = list(zip(latitude_trimmed.ravel(), longitude_trimmed.ravel(), rain_trimmed.ravel()))
            plugins.HeatMap(heatmap_data, radius=10, blur=20).add_to(m)

            # Display the map using an iframe
            st.write("Rainfall Heatmap")
            folium_map_html = m._repr_html_()
            st.write(folium_map_html, unsafe_allow_html=True)
        else:
            # Display the heatmap using Matplotlib
            aspect_ratio = (longitude_trimmed.max() - longitude_trimmed.min()) / (latitude_trimmed.max() - latitude_trimmed.min())
            fig = plt.figure(figsize=(6 * aspect_ratio, 6))
            plt.imshow(rain_trimmed, cmap='rainbow', extent=[longitude_trimmed.min(), longitude_trimmed.max(), latitude_trimmed.min(), latitude_trimmed.max()], vmax=5)
            plt.colorbar(label='mm/hr')
            plt.title('Rainfall_Pred_KMA')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            st.pyplot(fig)

    except Exception as e:
        st.write("Error during loading:", e)
    finally:
        os.remove(uploaded_file.name)  # Remove the temporary uploaded file

else:
    st.write("Please upload a NetCDF file using the sidebar.")
