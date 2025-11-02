import cv2
import mediapipe as mp
import numpy as np

class HandGestureMath:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.a = 7   # You can modify or integrate dynamic operand capture
        self.b = 3

    def count_fingers(self, landmarks):
        # Counts the number of extended fingers (basic method)
        fingers = 0
        tips_ids = [8, 12, 16, 20]  # IDs for fingertips: Index, Middle, Ring, Pinky
        for tip in tips_ids:
            if landmarks[tip].y < landmarks[tip - 2].y:  # opened finger is above the joint
                fingers += 1
        # Thumb: check if open to the right (for right hand, flip logic for left)
        if landmarks[4].x > landmarks[3].x:
            fingers += 1
        return fingers

    def get_operation(self, fingers):
        # Map finger count to math operation symbol and function
        operations = {
            2: ('+', lambda x, y: x + y),
            3: ('-', lambda x, y: x - y),
            4: ('*', lambda x, y: x * y),
            5: ('/', lambda x, y: x / y if y != 0 else 'Err')
        }
        return operations.get(fingers, (None, None))

    def run(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            expr, result = "--", "--"
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                fingers = self.count_fingers(hand_landmarks.landmark)
                op, func = self.get_operation(fingers)
                if op and func:
                    expr = f"{self.a} {op} {self.b}"
                    result = func(self.a, self.b)
            # Display results
            cv2.putText(frame, f"Expr: {expr}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f"Result: {result}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 0), 2)
            cv2.imshow('Hand Gesture Math', frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    HandGestureMath().run()
