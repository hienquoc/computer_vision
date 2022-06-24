import cv2
import numpy as np
from utility.utility import Utility


class Computer_vision:
    def __init__(self, user_selected_tracker_type=2):
        self.class_selected_tracker_type = user_selected_tracker_type
        self.video = None
        self.video_frame = None
        self.video_status = None
        self.debug = Utility()

    def setup_tracker(self):
        # Create a tracker object

        tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'CSRT', 'MOSSE']
        tracker_type = tracker_types[self.class_selected_tracker_type]

        # Tracker selector
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        elif tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        elif tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        elif tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        elif tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        elif tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        elif tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()
        elif tracker_type == "MOSSE":
            tracker = cv2.TrackerMOSSE_create()
        else:
            tracker = None
            print('Incorrect tracker name')
            print('Available trackers are:')
            for t in trackerTypes:
                print(t)

    def read_video(self, video_file_path):
        # Read video
        self.video = cv2.VideoCapture(video_file_path)

        # Exit if video not opened.
        if not self.video.isOpened():
            print("Could not open video")

    def read_frame(self):
        # Read first frame
        self.video_status, self.video_frame = self.video.read()
        self.debug.title = 'class: Computer_vision def: read_frame'
        self.debug_variable_dictionary = {'video_status': self.video_status,
                                          'video_frame': self.video_frame}
        self.debug.print_value_dictionary(self.debug_variable_dictionary)

    def object_tracker_main(self):
        # Initialize tracker
        self.setup_tracker()
        self.read_frame()
        # Initialize tracker with first frame and bounding box
