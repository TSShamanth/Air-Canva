import numpy as np
import cv2
import mediapipe as mp
from collections import deque
import logging
import signal
import sys

# Function to set trackbar values
def setValues(x):
    pass

# Signal handler for graceful termination
def signal_handler(sig, frame):
    logging.info("Exiting gracefully...")
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Function to run OpenCV functionality
def run_opencv():
    logging.info("Starting OpenCV...")
    # Create a window named 'Color detectors'
    cv2.namedWindow("Color detectors")

    # Create the trackbars
    cv2.createTrackbar("Upper Hue", "Color detectors", 153, 180, setValues)
    cv2.createTrackbar("Upper Saturation", "Color detectors", 255, 255, setValues)
    cv2.createTrackbar("Upper Value", "Color detectors", 255, 255, setValues)
    cv2.createTrackbar("Lower Hue", "Color detectors", 64, 180, setValues)
    cv2.createTrackbar("Lower Saturation", "Color detectors", 72, 255, setValues)
    cv2.createTrackbar("Lower Value", "Color detectors", 49, 255, setValues)

    # Initialize deque points for drawing
    bpoints = [deque(maxlen=512)]
    gpoints = [deque(maxlen=512)]
    rpoints = [deque(maxlen=512)]
    ypoints = [deque(maxlen=512)]

    # Color indices and default color
    blue_index = 0
    green_index = 0
    red_index = 0
    yellow_index = 0

    # Kernel for morphological operations
    kernel = np.ones((5, 5), np.uint8)

    # Colors and corresponding indices
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
    colorindex = 0

    # Create the paint window
    paintWindow = np.zeros((471, 636, 3)) + 255
    paintWindow = cv2.rectangle(paintWindow, (40, 1), (140, 65), (0, 0, 0), 2)
    paintWindow = cv2.rectangle(paintWindow, (160, 1), (255, 65), colors[0], -1)
    paintWindow = cv2.rectangle(paintWindow, (275, 1), (370, 65), colors[1], -1)
    paintWindow = cv2.rectangle(paintWindow, (390, 1), (485, 65), colors[2], -1)
    paintWindow = cv2.rectangle(paintWindow, (505, 1), (600, 65), colors[3], -1)

    cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 2, cv2.LINE_AA)
    cv2.namedWindow("Paint", cv2.WINDOW_AUTOSIZE)

    # Initialize MediaPipe hands module
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mpDraw = mp.solutions.drawing_utils

    # Start capturing video
    cap = cv2.VideoCapture(0)

    while True:
        # Reading the frame from the camera
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame to see the same side of yours
        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Get trackbar positions
        u_hue = cv2.getTrackbarPos("Upper Hue", "Color detectors")
        u_saturation = cv2.getTrackbarPos("Upper Saturation", "Color detectors")
        u_value = cv2.getTrackbarPos("Upper Value", "Color detectors")
        l_hue = cv2.getTrackbarPos("Lower Hue", "Color detectors")
        l_saturation = cv2.getTrackbarPos("Lower Saturation", "Color detectors")
        l_value = cv2.getTrackbarPos("Lower Value", "Color detectors")
        Upper_hsv = np.array([u_hue, u_saturation, u_value])
        Lower_hsv = np.array([l_hue, l_saturation, l_value])

        # Create color selection rectangles
        frame = cv2.rectangle(frame, (40, 1), (140, 65), (0, 0, 0), -1)
        frame = cv2.rectangle(frame, (160, 1), (255, 65), colors[0], -1)
        frame = cv2.rectangle(frame, (275, 1), (370, 65), colors[1], -1)
        frame = cv2.rectangle(frame, (390, 1), (485, 65), colors[2], -1)
        frame = cv2.rectangle(frame, (505, 1), (600, 65), colors[3], -1)
        cv2.putText(frame, "CLEAR ALL", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 2, cv2.LINE_AA)

        # Detect hand landmarks
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

                xList = []
                yList = []
                for lm in handLms.landmark:
                    h, w, c = frame.shape
                    x, y = int(lm.x * w), int(lm.y * h)
                    xList.append(x)
                    yList.append(y)

                index_tip = (xList[8], yList[8])  # Index finger tip
                index_dip = (xList[7], yList[7])  # Index finger DIP joint

                # Check if index finger is extended
                if index_tip and index_dip:
                    dist = np.sqrt((index_tip[0] - index_dip[0]) ** 2 + (index_tip[1] - index_dip[1]) ** 2)
                    if dist > 30:  # Adjust threshold as per your need
                        # Check if the index finger is within the bounds of the color buttons
                        if 40 <= index_tip[0] <= 140 and 1 <= index_tip[1] <= 65:
                            # Clear all points
                            bpoints = [deque(maxlen=512)]
                            gpoints = [deque(maxlen=512)]
                            rpoints = [deque(maxlen=512)]
                            ypoints = [deque(maxlen=512)]
                            paintWindow = np.zeros((471, 636, 3), dtype=np.uint8) + 255
                        elif 160 <= index_tip[0] <= 255 and 1 <= index_tip[1] <= 65:
                            colorindex = 0
                        elif 275 <= index_tip[0] <= 370 and 1 <= index_tip[1] <= 65:
                            colorindex = 1
                        elif 390 <= index_tip[0] <= 485 and 1 <= index_tip[1] <= 65:
                            colorindex = 2
                        elif 505 <= index_tip[0] <= 600 and 1 <= index_tip[1] <= 65:
                            colorindex = 3
                        else:
                            # Draw only when finger is extended and not near the color buttons
                            if colorindex == 0:
                                bpoints[blue_index].appendleft(index_tip)
                            elif colorindex == 1:
                                gpoints[green_index].appendleft(index_tip)
                            elif colorindex == 2:
                                rpoints[red_index].appendleft(index_tip)
                            elif colorindex == 3:
                                ypoints[yellow_index].appendleft(index_tip)

        # Append the points to the deque list
        points = [bpoints, gpoints, rpoints, ypoints]

        for i in range(len(points)):
            for j in range(len(points[i])):
                for k in range(1, len(points[i][j])):
                    if points[i][j][k - 1] is None or points[i][j][k] is None:
                        continue
                    cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                    cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

        # Display all the windows
        cv2.imshow("Tracking", frame)
        cv2.imshow("Paint", paintWindow)

        # Break the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_opencv()
