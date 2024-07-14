import cv2
import mediapipe as mp
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return math.sqrt((point2.x - point1.x)**2 + (point2.y - point1.y)**2)

# Function to calculate volume level based on number of fingers
def calculate_volume_level(num_fingers):
    # Calculate the volume level based on the number of fingers extended
    return min(20 * num_fingers, 100)  # Maximum volume is capped at 100

# MediaPipe hand detection initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# OpenCV camera initialization
cap = cv2.VideoCapture(0)

# Function to set system volume
def set_system_volume(volume):
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume_object = cast(interface, POINTER(IAudioEndpointVolume))
    volume_object.SetMasterVolumeLevelScalar(volume / 100, None)

prev_num_fingers = 0
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    # Process the image with MediaPipe Hands.
    results = hands.process(image)
    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    num_fingers = 0
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmarks for fingers
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            # Calculate distances between finger tips and base
            thumb_to_index = calculate_distance(thumb_tip, index_tip)
            index_to_middle = calculate_distance(index_tip, middle_tip)
            middle_to_ring = calculate_distance(middle_tip, ring_tip)
            ring_to_pinky = calculate_distance(ring_tip, pinky_tip)

            # Count the number of extended fingers for this hand
            num_fingers += sum([1 for d in [thumb_to_index, index_to_middle, middle_to_ring, ring_to_pinky] if d > 0.05])

    if num_fingers != prev_num_fingers:
        if num_fingers > 0:
            # Adjust system volume to 20 when fingers are extended
            set_system_volume(20)
        prev_num_fingers = num_fingers

    # Show the image with OpenCV
    cv2.imshow('Hand Tracking', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

hands.close()
cap.release()
cv2.destroyAllWindows()
