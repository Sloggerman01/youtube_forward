import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Capture video from the webcam
cap = cv2.VideoCapture(0)

# Finger landmark indices
index_tip = 8   # Index finger tip
thumb_tip = 4   # Thumb tip

# Gesture control flags
last_action_time = time.time()
cooldown = 0.5  # Cooldown period to avoid rapid triggering
is_paused = False  # Initial state for pause

def is_fist(lm_list):
    """Check if all fingers are closed (fist gesture)."""
    return all(lm_list[i].y > lm_list[i - 2].y + 0.1 for i in range(8, 21))

def is_tilt_left(lm_list):
    """Check if the index finger is tilted left."""
    return lm_list[index_tip].x < lm_list[thumb_tip].x

def is_tilt_right(lm_list):
    """Check if the index finger is tilted right."""
    return lm_list[index_tip].x > lm_list[thumb_tip].x

def is_neutral_position(lm_list):
    """Check if the hand is in a neutral position (fingers down)."""
    return all(lm_list[i].y > lm_list[i - 2].y + 0.1 for i in range(8, 21))

# Main loop for video capture and gesture detection
while True:
    ret, img = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    img = cv2.flip(img, 1)  # Flip the image for a mirror effect
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for MediaPipe processing
    results = hands.process(img_rgb)  # Process the image for hand landmarks

    current_time = time.time()  # Get current time for cooldown management

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            lm_list = [lm for lm in hand_landmark.landmark]  # List of landmarks for the current hand

            # Check for gestures if the cooldown period has passed
            if current_time - last_action_time > cooldown:
                if is_fist(lm_list):  # Pause/Play gesture
                    if not is_paused:
                        pyautogui.press('space')  # Trigger space for pause
                        print("Fist detected: Pause triggered")
                    is_paused = True  # Update pause state
                    last_action_time = current_time
                elif is_tilt_left(lm_list):  # Rewind gesture
                    pyautogui.press('left')
                    print("Index tilted left: Rewind triggered")
                    last_action_time = current_time
                elif is_tilt_right(lm_list):  # Forward gesture
                    pyautogui.press('right')
                    print("Index tilted right: Forward triggered")
                    last_action_time = current_time
                elif is_neutral_position(lm_list):  # In neutral position, reset last action time
                    last_action_time = current_time

    # Draw hand landmarks on the image
    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmark, mp_hands.HAND_CONNECTIONS)

    # Display the image with hand tracking
    cv2.imshow("Hand Tracking", img)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
