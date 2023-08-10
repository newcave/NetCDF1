import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import folium
import netCDF4 as nc
import os
import cartopy.crs as ccrs

# Create a Streamlit app
st.title('Rainfall Prediction - KMA')
st.sidebar.title('Upload NetCDF File')
uploaded_file = st.sidebar.file_uploader("Upload your NetCDF file", type=["nc"])

show_cartopy_map = st.sidebar.checkbox("Show Cartopy Map")

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

        if show_cartopy_map:
            # Calculate aspect ratio
            aspect_ratio = (longitude_trimmed.max() - longitude_trimmed.min()) / (latitude_trimmed.max() - latitude_trimmed.min())

            # Create a map using PlateCarree projection
            fig = plt.figure(figsize=(10, 10 / aspect_ratio))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

            # Plot the heatmap using pcolormesh
            heatmap = ax.pcolormesh(longitude_trimmed, latitude_trimmed, rain_trimmed, cmap='rainbow', vmax=5, transform=ccrs.PlateCarree())

            # Add colorbar
            cbar = plt.colorbar(heatmap, ax=ax, label='mm/hr', orientation='vertical')

            # Set map title and labels
            ax.set_title('Rainfall Prediction')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')

            # Add coastlines
            ax.coastlines()

            # Show the plot
            st.pyplot(fig)
            
            # Additional plot to the right
            aspect_ratio = (longitude_trimmed.max() - longitude_trimmed.min()) / (latitude_trimmed.max() - latitude_trimmed.min())
            fig2 = plt.figure(figsize=(6, 6))
            plt.imshow(rain_trimmed, cmap='rainbow', extent=[longitude_trimmed.min(), longitude_trimmed.max(), latitude_trimmed.min(), latitude_trimmed.max()], vmax=5, origin='lower')
            plt.colorbar(label='mm/hr', orientation='vertical')
            plt.title('Rainfall_Pred_KMA')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            
            cols = st.columns([1, 2])
            cols[0].pyplot(fig2)
            cols[1].write("fig added")
            
        else:
            # Display the heatmap using Matplotlib
            aspect_ratio = (longitude_trimmed.max() - longitude_trimmed.min()) / (latitude_trimmed.max() - latitude_trimmed.min())
            fig = plt.figure(figsize=(6 * aspect_ratio, 6))
            plt.imshow(rain_trimmed, cmap='rainbow', extent=[longitude_trimmed.min(), longitude_trimmed.max(), latitude_trimmed.min(), latitude_trimmed.max()], vmax=5, origin='lower')
            plt.colorbar(label='mm/hr', orientation='vertical')
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
