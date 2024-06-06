import cv2
import numpy as np
import time

# List to store the detected centers of the concentric circles with timestamps
centers = []

# Open video file
cap = cv2.VideoCapture("humanoidTask.mp4")

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 10, (1920, 1080))


def predict_next_center(centers):
    if len(centers) < 2:
        return None
    # Calculate velocities
    velocities = []
    for i in range(1, len(centers)):
        dx = centers[i][0][0] - centers[i - 1][0][0]
        dy = centers[i][0][1] - centers[i - 1][0][1]
        dt = centers[i][1] - centers[i - 1][1]
        if dt > 0:
            velocities.append((dx / dt, dy / dt))
    if not velocities:
        return None
    # Average velocity
    avg_velocity = np.mean(velocities, axis=0)
    # Predict next center
    last_center, last_time = centers[-1]
    predicted_center = (
    int(last_center[0] + avg_velocity[0] * (1 / 10)), int(last_center[1] + avg_velocity[1] * (1 / 10)))
    return predicted_center


while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()

    # Check if frame read successfully
    if not ret:
        print("Reached end of the video or failed to read frame.")
        break

    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for each color
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    lower_blue = np.array([90, 100, 100])
    upper_blue = np.array([120, 255, 255])

    # Create masks for each color
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Find the center of the largest contour for each color
    detected_centers = []
    for mask in [mask_red, mask_yellow, mask_blue]:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            if radius > 10:  # Filter out small circles that may be noise
                center = (int(x), int(y))
                detected_centers.append(center)
                cv2.circle(frame, center, 10, (0, 255, 0), -1)
                centers.append((center, time.time()))

    # Predict the center in the next frame
    predicted_center = predict_next_center(centers)
    if predicted_center:
        cv2.circle(frame, predicted_center, 10, (255, 0, 0), -1)

    # Draw circles at each detected center for trajectory
    for center, _ in centers:
        cv2.circle(frame, center, 5, (0, 0, 255), -1)

    # Show the frame with detected circles and trajectory
    cv2.imshow('Frame', frame)

    # Write the frame with annotations to the output video
    out.write(frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release VideoCapture and VideoWriter objects
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()






