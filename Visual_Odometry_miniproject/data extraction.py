import pandas as pd
import utm
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv('Data/DJIFlightRecord_2021-03-18_[13-04-51]-TxtLogToCsv.csv', encoding='latin1')

# Find the indices where recording starts and stops
recording_start_indices = df.index[df['CUSTOM.isVideo'] == 'Recording'].tolist()
recording_stop_indices = df.index[df['CUSTOM.isVideo'] == 'Stop'].tolist()

# Filter data for video 1
video1_start = recording_start_indices[0]
video1_stop = recording_stop_indices[0]
video1_data = df.iloc[video1_start:video1_stop+1]

# Extract latitude and longitude columns
latitude = video1_data['OSD.latitude'].tolist()
longitude = video1_data['OSD.longitude'].tolist()

# Convert latitude and longitude to UTM coordinates
utm_coords = [utm.from_latlon(lat, lon) for lat, lon in zip(latitude, longitude)]
utm_easting = [coord[0] for coord in utm_coords]
utm_northing = [coord[1] for coord in utm_coords]

# Plot the UTM coordinates
plt.figure(figsize=(10, 6))
plt.scatter(utm_easting, utm_northing, s=5, c='blue', label='Video 1 GPS Data')
plt.xlabel('UTM Easting (m)')
plt.ylabel('UTM Northing (m)')
plt.title('GPS Data from Video 1 in UTM Coordinates')
plt.grid(True)
plt.legend()

# Save the figure as an image file
plt.savefig('Outputs/gps_data_plot.png')

# Show the plot
plt.show()
