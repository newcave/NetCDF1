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
st.title('Rainfall Prediction - KPPM_KMA')
st.sidebar.title('Upload NetCDF File')
uploaded_file = st.sidebar.file_uploader("Upload your NetCDF file", type=["nc"])
use_default_file = st.sidebar.checkbox("Use Existing Files in Server")

if use_default_file:
    default_file_path = './data/RN_KMA_NetCDF_2023081421.NC'
    default_file_path2 = './data/RN_KMA_NetCDF_2023081400.NC'
    uploaded_file_name = st.sidebar.selectbox("Select Default File", [default_file_path, default_file_path2])
    st.sidebar.write(f"Using Default File: {os.path.basename(uploaded_file_name)}")

else:
    if uploaded_file is not None:
        # Load NetCDF data
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.read())
        uploaded_file_name = uploaded_file.name
    else:
        uploaded_file_name = None

if uploaded_file_name is not None:
    try:
        df = nc.Dataset(uploaded_file_name)
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

        # Check if the checkbox for Cartopy map display is checked
        show_cartopy_map = st.sidebar.checkbox("Un-check for larger Images", value=True)

        if show_cartopy_map:
            aspect_ratio = (longitude_trimmed.max() - longitude_trimmed.min()) / (latitude_trimmed.max() - latitude_trimmed.min())

            # Create a map using PlateCarree projection
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7 / aspect_ratio), subplot_kw={'projection': ccrs.PlateCarree()})

            # Plot the heatmap using pcolormesh on the first subplot
            heatmap = ax1.pcolormesh(longitude_trimmed, latitude_trimmed, rain_trimmed, cmap='rainbow', vmax=5, transform=ccrs.PlateCarree())
            cbar = plt.colorbar(heatmap, ax=ax1, label='mm/hr', orientation='vertical')
            ax1.set_title('Rainfall Prediction (Cartopy)')
            ax1.set_xlabel('Longitude')
            ax1.set_ylabel('Latitude')
            ax1.coastlines()

            # Plot the heatmap using imshow on the second subplot
            heatmap2 = ax2.imshow(rain_trimmed, cmap='rainbow', extent=[longitude_trimmed.min(), longitude_trimmed.max(), latitude_trimmed.min(), latitude_trimmed.max()], vmax=5, origin='lower')
            cbar2 = plt.colorbar(heatmap2, ax=ax2, label='mm/hr', orientation='vertical')
            ax2.set_title('Rainfall Prediction')
            ax2.set_xlabel('Longitude')
            ax2.set_ylabel('Latitude')
            ax2.set_xticks(np.linspace(longitude_trimmed.min(), longitude_trimmed.max(), num=5))
            ax2.set_yticks(np.linspace(latitude_trimmed.min(), latitude_trimmed.max(), num=5))

            # Display the plots using Streamlit
            st.pyplot(fig)

        else:
            fig2, ax1 = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
            heatmap = ax1.pcolormesh(longitude_trimmed, latitude_trimmed, rain_trimmed, cmap='rainbow', vmax=5, transform=ccrs.PlateCarree())
            cbar = plt.colorbar(heatmap, ax=ax1, label='mm/hr', orientation='vertical')
            ax1.set_xlabel('Longitude')
            ax1.set_ylabel('Latitude')
            ax1.coastlines()

            xticks = np.arange(longitude_trimmed.min(), longitude_trimmed.max() + 1, 2)
            yticks = np.arange(latitude_trimmed.min(), latitude_trimmed.max() + 1, 2)
            ax1.set_xticks(xticks, crs=ccrs.PlateCarree())
            ax1.set_yticks(yticks, crs=ccrs.PlateCarree())
            ax1.xaxis.set_major_formatter(plt.FixedFormatter(np.abs(xticks)))
            ax1.yaxis.set_major_formatter(plt.FixedFormatter(np.abs(yticks)))
            xticklabels = ['{:.3f}'.format(x) for x in xticks]
            yticklabels = ['{:.3f}'.format(y) for y in yticks]
            ax1.set_xticklabels(xticklabels)
            ax1.set_yticklabels(yticklabels)
            st.pyplot(fig2)

    except Exception as e:
        st.write("Error during loading:", e)
    finally:
        if not use_default_file:
            os.remove(uploaded_file_name)  # Remove the temporary uploaded file

else:
    st.write("Please upload a NetCDF file using the sidebar.")
