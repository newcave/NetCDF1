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
