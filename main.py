import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

positions = []       # For temporal smoothing
trail_positions = [] # To store positions for trail drawing
trail_image = None   # To accumulate the trail over time
frame_count = 0      # To keep track of frame numbers

width = 1000  # Example value, replace with your camera's max width
height = 1080  # Example value, replace with your camera's max height

cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    frame = cv.flip(frame,1)

    frame_count += 1  # Increment frame counter

    # Initialize trail_image after getting the frame dimensions
    if trail_image is None:
        trail_image = np.zeros_like(frame).astype(np.float32)  # Use float32 for precision during fading

    # Convert to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    gray_blurred = cv.GaussianBlur(gray, (5, 5), 0)

    # Apply Thresholding to get binary image
    _, thresh = cv.threshold(gray_blurred, 200, 255, cv.THRESH_BINARY)

    # Apply Morphological Operations
    kernel = np.ones((5, 5), np.uint8)
    thresh_morph = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
    thresh_morph = cv.morphologyEx(thresh_morph, cv.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv.findContours(thresh_morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour
        max_contour = max(contours, key=cv.contourArea)
        x, y, w, h = cv.boundingRect(max_contour)

        # Update positions for smoothing
        center = (x + w // 2, y + h // 2)
        positions.append(center)
        N = 5  # Number of frames to average over
        if len(positions) > N:
            positions.pop(0)

        # Compute average position
        avg_x = int(sum([p[0] for p in positions]) / len(positions))
        avg_y = int(sum([p[1] for p in positions]) / len(positions))

        # Update trail positions
        trail_positions.append((avg_x, avg_y))
        max_length = 50  # Maximum length of the trail
        if len(trail_positions) > max_length:
            trail_positions.pop(0)

        # Fade the trail image to create a fading effect
        trail_image *= 0.9  # Decay factor (adjust between 0 and 1 for faster or slower fading)

        # Draw the trail on the trail_image
        for i in range(1, len(trail_positions)):
            cv.line(trail_image, trail_positions[i - 1], trail_positions[i], (0, 0, 255), 2)

        # Draw rectangle at the smoothed position on the original frame
        cv.rectangle(frame, (avg_x - w // 2, avg_y - h // 2), (avg_x + w // 2, avg_y + h // 2), (0, 255, 0), 2)

        # Draw the same rectangle on the thresholded image for comparison
        thresh_color = cv.cvtColor(thresh_morph, cv.COLOR_GRAY2BGR)
        cv.rectangle(thresh_color, (avg_x - w // 2, avg_y - h // 2), (avg_x + w // 2, avg_y + h // 2), (0, 255, 0), 2)

    else:
        # If no contours are found, you might want to handle this case
        thresh_color = cv.cvtColor(thresh_morph, cv.COLOR_GRAY2BGR)
        pass

    # Combine the original frame with the trail_image
    combined_frame = cv.addWeighted(frame.astype(np.float32), 1.0, trail_image, 1.0, 0.0)
    combined_frame = np.clip(combined_frame, 0, 255).astype(np.uint8)  # Ensure values are within valid color range

    # Optionally, add text information on the combined frame
    cv.putText(combined_frame, f'Frame: {frame_count}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Stack the combined frame and the thresholded image side by side
    combined_display = np.hstack((combined_frame, thresh_color))

    # Display the combined image
    cv.imshow('Original Frame with Trail and Processed Image', combined_display)

    k = cv.waitKey(30) & 0xff
    if k == 27:  # Press 'ESC' to exit
        break

cap.release()
cv.destroyAllWindows()