"""
hand_detector.py

Part of the GestureSpeak AI project.

This module provides a HandDetector class that wraps Google's MediaPipe
Hands solution for detecting hands and drawing landmarks on OpenCV frames.

This file intentionally contains ONLY hand detection logic.
It does NOT contain any Flask routes/app code and does NOT contain
any webcam capture (cv2.VideoCapture) code. It is meant to be imported
and used by other parts of the GestureSpeak AI project.
"""

import cv2
import mediapipe as mp


class HandDetector:
    """
    A wrapper class around MediaPipe Hands that detects hands in a given
    OpenCV (BGR) frame, draws the hand landmarks/connections on it, and
    returns the processed frame.
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """
        Initialize the MediaPipe Hands model and drawing utilities.

        Args:
            static_image_mode (bool): If False, treats input images as a
                video stream (enables tracking between frames for better
                performance). Set True for independent, unrelated images.
            max_num_hands (int): Maximum number of hands to detect.
            min_detection_confidence (float): Minimum confidence value
                ([0.0, 1.0]) for hand detection to be considered successful.
            min_tracking_confidence (float): Minimum confidence value
                ([0.0, 1.0]) for the hand landmarks to be considered
                tracked successfully.
        """
        # MediaPipe Hands solution module
        self.mp_hands = mp.solutions.hands

        # Create the Hands detector instance with the given configuration
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        # MediaPipe drawing utilities (used to draw landmarks/connections)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def detect_hands(self, frame):
        """
        Detect hands in the given frame, draw landmarks on any detected
        hands, and return the processed frame.

        Args:
            frame: An OpenCV frame (BGR numpy array) to process.

        Returns:
            The processed OpenCV frame (BGR numpy array) with hand
            landmarks and connections drawn on it, if any hands were
            detected. If no hands are detected, the original frame is
            returned unmodified.
        """
        # MediaPipe expects RGB images, but OpenCV frames are in BGR,
        # so we convert the color space before processing.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run hand detection/tracking on the RGB frame
        results = self.hands.process(rgb_frame)

        # If one or more hands were detected, draw their landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,  # Draw directly on the original BGR frame
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style(),
                )

        # Return the processed frame (with or without drawn landmarks)
        return frame, results