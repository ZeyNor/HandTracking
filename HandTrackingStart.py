import cv2
import mediapipe as mp
import time

# Initialize the webcam capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Initialize the Mediapipe Hands module
mpHands = mp.solutions.hands
hands = mpHands.Hands()

# Initialize the Mediapipe drawing utilities
mpDraw = mp.solutions.drawing_utils

# Initialize variables to track time for calculating frames per second (fps)
pTime = 0
cTime = 0

# Main loop to process frames from the webcam
while True:
    # Read a frame from the webcam
    success, img = cap.read()

    # Convert the frame from BGR to RGB color format (required by Mediapipe)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands and landmarks
    results = hands.process(imgRGB)

    # Check if hands are detected in the frame
    if results.multi_hand_landmarks:
        # Iterate through each detected hand
        for handLms in results.multi_hand_landmarks:
            # Iterate through each landmark point in the hand
            for id, lm in enumerate(handLms.landmark):
                # Get the dimensions of the image
                h, w, c = img.shape
                # Calculate the pixel coordinates of the landmark
                cx, cy = int(lm.x * w), int(lm.y * h)
                # Draw a circle at the landmark's position
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            # Draw connections between the landmarks to form the hand skeleton
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # Calculate frames per second (fps)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Display the calculated fps on the image
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # Display the annotated image in a window
    cv2.imshow("Image", img)

    # Wait for a key press and exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()


