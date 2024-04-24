import numpy as np
import cv2
from Map import Map
from TrackedCamera import TrackedCamera
from TrackedPoint import TrackedPoint
from Observation import Observation
import xml.etree.ElementTree as ET

# ----------------------------------------------------------------- Map Init ------------------------------------------------------------------

# Initialize map
map = Map()

# Paths to your image files and load images
frame1_path = 'Extracted frames/frame_1225.jpg'
frame2_path = 'Extracted frames/frame_1250.jpg'
frame1 = cv2.imread(frame1_path)
frame2 = cv2.imread(frame2_path)

# Initialize ORB detector and detect keypoints and descriptors
orb = cv2.ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(frame1, None)
keypoints2, descriptors2 = orb.detectAndCompute(frame2, None)

# Draw and display keypoints on the images
frame1_with_keypoints = cv2.drawKeypoints(frame1, keypoints1, None, color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
frame2_with_keypoints = cv2.drawKeypoints(frame2, keypoints2, None, color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
resized_frame1 = cv2.resize(frame1_with_keypoints, (1080, 720))
resized_frame2 = cv2.resize(frame2_with_keypoints, (1080, 720))
cv2.imshow('Keypoints on Frame 1225', resized_frame1)
cv2.imshow('Keypoints on Frame 1250', resized_frame2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the images to files
cv2.imwrite('Outputs/frame1225_with_features.jpg', frame1_with_keypoints)
cv2.imwrite('Outputs/frame1250_with_features.jpg', frame2_with_keypoints)

# Feature matching and display
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance)
matched_image = cv2.drawMatches(frame1, keypoints1, frame2, keypoints2, matches[:], None, flags=2)
resized_matched_image = cv2.resize(matched_image, (2560, 1440))
cv2.imshow('Matches', resized_matched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('Outputs/matched_image.jpg', matched_image)

# Camera intrinsic and distortion coefficients from XML
tree = ET.parse('Data/phantom4pro-calibration.xml')
root = tree.getroot()
f = float(root.find('f').text)
cx = float(root.find('cx').text)
cy = float(root.find('cy').text)
K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
dist_coeffs = np.array([float(root.find('k1').text), float(root.find('k2').text),
                        float(root.find('p1').text), float(root.find('p2').text), float(root.find('k3').text)])

# Points from matches and undistort points
points1 = np.array([keypoints1[m.queryIdx].pt for m in matches], dtype=np.float32)
points2 = np.array([keypoints2[m.trainIdx].pt for m in matches], dtype=np.float32)
points1 = cv2.undistortPoints(np.expand_dims(points1, axis=1), K, dist_coeffs)
points2 = cv2.undistortPoints(np.expand_dims(points2, axis=1), K, dist_coeffs)

# Estimating essential matrix
E, mask = cv2.findEssentialMat(points1, points2, K, cv2.RANSAC, 0.999, 1.0)

# First, we need the fundamental matrix from the essential matrix
F = cv2.findFundamentalMat(points1, points2, method=cv2.FM_LMEDS)[0]

# Calculate the epipolar lines for points in the second image
# lines1 for points in image 1 with respect to points in image 2
# lines2 for points in image 2 with respect to points in image 1
lines1 = cv2.computeCorrespondEpilines(points2.reshape(-1, 1, 2), 2, F)
lines2 = cv2.computeCorrespondEpilines(points1.reshape(-1, 1, 2), 1, F)
lines1 = lines1.reshape(-1, 3)
lines2 = lines2.reshape(-1, 3)

def distance_to_line(point, line):
    # Point expected to be of shape (1, 2)
    a, b, c = line
    x0, y0 = point[0][0], point[0][1]  # Adjusting indexing to match the shape from undistortPoints
    return np.abs(a * x0 + b * y0 + c) / np.sqrt(a**2 + b**2)

# Calculate distances with corrected point access
distances1 = np.array([distance_to_line(points1[i], lines1[i]) for i in range(len(points1))])
distances2 = np.array([distance_to_line(points2[i], lines2[i]) for i in range(len(points2))])

# Summary statistics
mean_distance1 = np.mean(distances1)
std_distance1 = np.std(distances1)
mean_distance2 = np.mean(distances2)
std_distance2 = np.std(distances2)

print(f"Statistics for Image 1 -> Image 2:")
print(f"Mean Distance: {mean_distance1}")
print(f"Standard Deviation: {std_distance1}")

print(f"Statistics for Image 2 -> Image 1:")
print(f"Mean Distance: {mean_distance2}")
print(f"Standard Deviation: {std_distance2}")

# Finding and print the recovered rotation and translation
_, R, t, mask = cv2.recoverPose(E, points1, points2, K)
print("Recovered rotation:")
print(R)
print("Recovered translation:")
print(t)


# ----------------------------------------------------------------- 3D Map Init ------------------------------------------------------------------
# Camera Projection Matrices
P1 = np.dot(K, np.hstack((np.eye(3), np.zeros((3, 1)))))  # Projection matrix for the first camera
P2 = np.dot(K, np.hstack((R, t)))  # Projection matrix for the second camera

# Triangulate points to get 3D coordinates in homogeneous coordinates
points_3d_hom = cv2.triangulatePoints(P1, P2, points1, points2)

# Convert homogeneous coordinates to 3D coordinates
points_3d = points_3d_hom[:3] / points_3d_hom[3]
    
# Add cameras to Map object
camera1 = TrackedCamera(np.eye(3), np.zeros(3), frame_id=1, frame=frame1)
camera2 = TrackedCamera(R, t, frame_id=2, frame=frame2)
camera1 = map.add_camera(camera1)
camera2 = map.add_camera(camera2)

# Add points and observations to the map
for i in range(points_3d.shape[1]):
    # Create a new TrackedPoint for the 3D coordinate
    x, y, z = points_3d[:, i]
    tracked_point = TrackedPoint(point=np.array([x, y, z]), descriptor=None, color=None, feature_id=(i, 'frame1-frame2'))
    map_point = map.add_point(tracked_point)

    # Add observations for each camera
    obs1 = Observation(point_id=map_point.point_id, camera_id=camera1.camera_id, image_coordinates=points1[i].reshape(-1))
    obs2 = Observation(point_id=map_point.point_id, camera_id=camera2.camera_id, image_coordinates=points2[i].reshape(-1))
    map.observations.append(obs1)
    map.observations.append(obs2)

# Calculating reprojection error of the current state of the map
print("Reprojection error of the current state of the map:")    
map.show_total_reprojection_error()

# Optimizing map using bundle adjustment
map.optimize_map()
print("Reprojection error post bundle adjustment:")    
map.show_total_reprojection_error()
