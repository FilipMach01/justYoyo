import cv2 as cv

cap = cv.VideoCapture(0)

# Set the resolution to maximum supported values
# Replace 'width' and 'height' with your camera's maximum resolution
width = 1920  # Example value, replace with your camera's max width
height = 1080  # Example value, replace with your camera's max height

cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)

# Confirm the resolution
actual_width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
print(f"Camera resolution set to: {actual_width} x {actual_height}")