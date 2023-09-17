import cv2

# Initialize the camera
cap = cv2.VideoCapture(0)  # 0 represents the default camera (usually the built-in webcam)

# Initialize variables to store the previous frame and the current frame
prev_frame = None

while True:
    # Capture the current frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale for movement detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_frame is not None:
        # Calculate the absolute difference between the current and previous frames
        frame_diff = cv2.absdiff(prev_frame, gray)

        # Apply a threshold to the difference image to identify areas with significant changes
        _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Iterate through the detected contours
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                # If the contour area is larger than a threshold (e.g., 500 pixels), consider it as movement
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Update the previous frame
    prev_frame = gray

    # Display the current frame
    cv2.imshow('Movement Detection', frame)

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
