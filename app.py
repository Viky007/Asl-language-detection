import cv2
import numpy as np
import pyttsx3
import time
from tensorflow.keras.models import load_model
import mediapipe as mp
import os

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize Text-to-Speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# ASL and gesture labels
asl_labels = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
    'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z', 'space', 'nothing', 'del'
]

def text_to_speech(text):
    engine.say(text)
    engine.runAndWait()

def preprocess_asl_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    normalized = resized.astype('float32') / 255.0
    reshaped = normalized.reshape(1, 28, 28, 1)
    return reshaped

def detect_hand_gesture(landmarks):
    thumb_tip_y = landmarks[mp_hands.HandLandmark.THUMB_TIP].y
    thumb_ip_y = landmarks[mp_hands.HandLandmark.THUMB_IP].y
    wrist_y = landmarks[mp_hands.HandLandmark.WRIST].y
    index_y = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    middle_y = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    ring_y = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y
    pinky_y = landmarks[mp_hands.HandLandmark.PINKY_TIP].y

    if (thumb_tip_y < wrist_y and
        index_y > wrist_y and
        middle_y > wrist_y and
        ring_y > wrist_y and
        pinky_y > wrist_y):
        return "Thumbs Up"
    elif (thumb_tip_y < wrist_y and index_y < wrist_y and middle_y < wrist_y and
          ring_y < wrist_y and pinky_y < wrist_y):
        return "Open Palm"
    elif (thumb_tip_y > wrist_y and index_y > wrist_y and middle_y > wrist_y and
          ring_y > wrist_y and pinky_y > wrist_y):
        return "Fist"
    else:
        return "Unknown Gesture"

def main():
    print("Loading ASL model...")
    asl_model_path = r'C:\Users\vigne\PycharmProjects\asl language detection\asl_cnn_model2.keras'

    try:
        asl_model = load_model(asl_model_path)
    except Exception as e:
        print("Error loading model:", e)
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    output_file = open("asl_output.txt", "w")
    recognized_text = ""
    last_pred_label = ""
    last_prediction_time = 0
    cooldown = 5.0

    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
        print("Starting combined ASL and gesture recognition. Press ESC to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                break

            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            asl_roi = frame[50:350, 50:350]  # ASL ROI
            asl_input = preprocess_asl_frame(asl_roi)
            asl_pred = asl_model.predict(asl_input)
            asl_index = np.argmax(asl_pred)
            asl_conf = asl_pred[0][asl_index]
            asl_label = asl_labels[asl_index]

            # MediaPipe gesture recognition
            gesture = "No Hand Detected"
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    gesture = detect_hand_gesture(hand_landmarks.landmark)

            # Draw ASL ROI and predictions
            cv2.rectangle(frame, (50, 50), (350, 350), (0, 255, 0), 2)
            cv2.putText(frame, f"ASL: {asl_label} ({asl_conf*100:.1f}%)", (50, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Show gesture on screen
            cv2.putText(frame, f"Gesture: {gesture}", (10, 420),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Update recognized text if confident
            current_time = time.time()
            if asl_conf > 0.8 and asl_label != last_pred_label and current_time - last_prediction_time > cooldown:
                if asl_label == 'space':
                    recognized_text += ' '
                elif asl_label == 'del':
                    recognized_text = recognized_text[:-1]
                elif asl_label != 'nothing':
                    recognized_text += asl_label

                output_file.seek(0)
                output_file.write(recognized_text)
                output_file.truncate()
                output_file.flush()
                text_to_speech(asl_label)

                last_pred_label = asl_label
                last_prediction_time = current_time

            # Optionally speak gesture
            if gesture != "Unknown Gesture" and gesture != last_pred_label and current_time - last_prediction_time > cooldown:
                text_to_speech(gesture)
                last_pred_label = gesture
                last_prediction_time = current_time

            cv2.imshow('ASL + MediaPipe Gesture Recognition', frame)
            if cv2.waitKey(5) & 0xFF == 27:  # ESC key
                break

    cap.release()
    output_file.close()
    cv2.destroyAllWindows()
    os.system("notepad asl_output.txt")

if __name__ == "__main__":
    main()
