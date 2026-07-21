"""
gesture_classifier.py

Gesture classification module for GestureSpeak AI.

This module receives hand landmarks from MediaPipe and determines
which gesture the user is showing.

Initially, this is a placeholder implementation. We'll gradually
add support for Indian Sign Language gestures.
"""


class GestureClassifier:
    """
    Class responsible for recognizing gestures from hand landmarks.
    """

    def __init__(self):
        pass

    def classify(self, hand_landmarks):
        """
        Classify the detected hand gesture.

        Args:
            hand_landmarks:
                MediaPipe hand landmarks.

        Returns:
            str:
                Name of the detected gesture.
        """

        if hand_landmarks is None:
            return "No Hand"

        # Placeholder.
        # We'll replace this with real gesture recognition logic.
        return "Hand Detected"