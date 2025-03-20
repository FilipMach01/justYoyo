import cv2
import numpy as np
from collections import deque
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import joblib
import time


# Define the generate_training_data function
def generate_training_data():
    # Generate random training data
    X_train = np.random.rand(100, 2)
    y_train = np.random.rand(100, 2)

    # Simulate movement prediction
    X_train[:, 0] += np.sin(np.arange(100))
    X_train[:, 1] += np.cos(np.arange(100))

    return X_train, y_train


# Set environment variables before importing OpenCV
import os

os.environ['OPENCV_LOG_LEVEL'] = 'debug'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '1'

# Initialize variables
cap = cv2.VideoCapture(0)
positions = deque(maxlen=50)  # Store last 50 positions
frame_count = 0
width, height = 640, 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Create a default model if none exists
model_path = 'movement_predictor.joblib'
if not os.path.exists(model_path):
    print(f"No pre-trained model found. Creating a default model...")
    X_train, y_train = generate_training_data()
    model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
else:
    print(f"Loading pre-trained model from {model_path}")
    model = joblib.load(model_path)


# Function to preprocess frame
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    return cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((5, 5)))


# Main tracking loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    frame_count += 1
    processed_frame = preprocess_frame(frame)

    # Find contours
    contours, _ = cv2.findContours(processed_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)

        # Calculate center
        center = (x + w // 2, y + h // 2)

        # Add to positions deque
        positions.append(center)

        # Smooth positions
        smoothed_center = tuple(map(lambda pair: ((pair[0] + pair[1]) // 2), positions))

        # Check if smoothed_center exists and has the expected length
        if len(smoothed_center) == 2:
            # Predict next position
            if len(positions) > 10:
                prev_positions = np.array(positions)[-10:]
                scaler = StandardScaler()
                scaled_positions = scaler.fit_transform(prev_positions)
                pred_next_pos = model.predict(scaled_positions.reshape(-1, 2))

                # Adjust prediction based on current velocity
                curr_velocity = np.linalg.norm(np.diff(positions[-2:])[:2])
                pred_next_pos *= 1.1 if curr_velocity < 5 else 1.5

                predicted_center = tuple(pred_next_pos.astype(int))

            # Draw rectangles
            cv2.rectangle(frame, (smoothed_center[0] - w // 2, smoothed_center[1] - h // 2),
                          (smoothed_center[0] + w // 2, smoothed_center[1] + h // 2),
                          (0, 255, 0), 2)

            # Optional: Draw predicted position
            if 'predicted_center' in locals():
                cv2.circle(frame, predicted_center, 5, (0, 0, 255), -1)
        else:
            print("Warning: smoothed_center does not exist or is empty")

    # Display frame
    cv2.imshow('Tracking', frame)

    # Exit on press 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
