import cv2
import os

# Path to the input video file
input_video_path = 'Data/DJI_0199.MOV'

# Create a directory to store the extracted frames
output_dir = 'Extracted frames'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Open the video file
cap = cv2.VideoCapture(input_video_path)

# Initialize frame count
frame_count = 0

# Loop through each frame in the video
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Increment frame count
    frame_count += 1

    # Skip the first 1200 frames
    if frame_count <= 1200:
        continue

    # Save every 25th frame
    if frame_count % 25 == 0:
        # Define the filename for the extracted frame
        frame_filename = os.path.join(output_dir, f'frame_{frame_count}.jpg')

        # Save the frame to disk
        cv2.imwrite(frame_filename, frame)

# Release the video capture object
cap.release()

print("Extraction complete.")

