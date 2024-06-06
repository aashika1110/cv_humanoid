import cv2
import numpy as np

# Load the video
cap = cv2.VideoCapture("bottlefliphand.mp4")

# Get the video's frame width, height, and frames per second
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object for the output video
out = cv2.VideoWriter('output_bottlefliphand.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Parameters for color thresholding (HSV range for blue cap)
lower_blue = np.array([90, 100, 100])
upper_blue = np.array([120, 255, 255])

# Minimum area for contour detection
min_contour_area = 100

# Height difference threshold for successful flip
success_height_threshold = 10  # Adjust as needed

# Initialize variables for tracking
start_cap_center = None
end_cap_center = None
flipped = False

frame_count = 0

while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()
    frame_count += 1
    if not ret:
        break

    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 1. Detect the bottle cap
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)

        # Filter contours by area
        if area > min_contour_area:
            # Get bounding rectangle of the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Draw a rectangle around the detected cap
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Calculate center of the rectangle
            cap_center = (x + w // 2, y + h // 2)

            # Store the cap center at the start and end of the video
            if frame_count == 1:
                start_cap_center = cap_center
            elif frame_count == int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
                end_cap_center = cap_center

    # Write the frame with annotations to the output video
    out.write(frame)

    # Display the frame with annotations
    cv2.imshow('Frame', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Check if the bottle successfully flipped based on the height difference
if start_cap_center and end_cap_center:
    height_diff = abs(start_cap_center[1] - end_cap_center[1])
    if height_diff < success_height_threshold:
        flipped = True

# Print result
if flipped:
    print("Bottle successfully flipped!")
else:
    print("Bottle not flipped.")

# Release VideoCapture and VideoWriter objects
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

























































