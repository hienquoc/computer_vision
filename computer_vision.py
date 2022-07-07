import cv2
import numpy as np
from utility.utility import Utility
import os
import sys
import os.path


class Computer_vision:
    def __init__(self, user_selected_tracker_type=2):
        self.class_selected_tracker_type = user_selected_tracker_type
        self.video = None
        self.video_frame = None
        self.video_status = None
        self.video_file_path = None
        self.debug = Utility()
        self.method_selector_dictionary = {}
        self.computer_vision_attributes = {}

    def method_selector(self):
        for key in self.method_selector_dictionary:
            if False:
                pass
            elif key == 'setup_tracker':
                self.setup_tracker()
            elif key == 'read_video':
                self.read_video()   # Read video
            elif key == 'read_frame':
                self.read_frame()
            elif key == 'object_tracker_main':
                self.object_tracker_main()
            elif key == 'yolo_object_dectection':
                self.yolo_object_dectection()
            elif key == 'getOutputsNames':
                self.getOutputsNames()
            elif key == 'method_selector':
                self.method_selector()

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

    def read_video(self):
        # Read video

        self.video = cv2.VideoCapture(self.video_file_path)
        self.debug.title = 'class: Computer_vision def: read_video'
        self.debug.debug_variable_dictionary = {'video_file_path': self.video_file_path}
        self.debug.print_value_dictionary()
        # Exit if video not opened.
        if not self.video.isOpened():
            print("Could not open video")

    def read_frame(self):
        # Read first frame
        while True:
            self.video_status, self.video_frame = self.video.read()
            self.debug.title = 'class: Computer_vision def: read_frame'
            self.debug.debug_variable_dictionary = {'video_status': self.video_status,
                                                    'video_frame': self.video_frame}
            self.method_selector()
            # self.debug.print_value_dictionary()
            cv2.imshow('Video', self.video_frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    def object_tracker_main(self):
        # Initialize tracker
        computer_vision_class_file_path = os.path.abspath(os.getcwd())
        self.video_file_path = computer_vision_class_file_path + '\soccer-ball.mp4'
        self.setup_tracker()
        self.read_video()
        self.read_frame()
        # Initialize tracker with first frame and bounding box

    def yolo_object_dectection(self):
        # Initialize the parameters
        objectnessThreshold = 0.5  # Objectness threshold
        confThreshold = 0.5  # Confidence threshold
        nmsThreshold = 0.4  # Non-maximum suppression threshold
        inpWidth = 416  # Width of network's input image
        inpHeight = 416  # Height of network's input image

        # Load names of classes
        computer_vision_class_file_path = os.path.abspath(os.getcwd())
        classesFile = computer_vision_class_file_path + "\coco.names"
        classes = None
        with open(classesFile, 'rt') as f:
            classes = f.read().rstrip('\n').split('\n')

        # Give the configuration and weight files for the model and load the network using them.
        modelConfiguration = computer_vision_class_file_path + "\yolov3.cfg"
        modelWeights = computer_vision_class_file_path + "\yolov3.weights"

        net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

    # Get the names of the output layers
    def getOutputsNames(self, net):
        # Get the names of all the layers in the network
        layersNames = net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Draw the predicted bounding box
    def drawPred(self, classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

        label = '%.2f' % conf

        # Get the label for the class name and its confidence
        if classes:
            assert (classId < len(classes))
            label = '%s:%s' % (classes[classId], label)

        # Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv2.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
                      (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

    # Remove the bounding boxes with low confidence using non-maxima suppression
    def postprocess(self, frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        classIds = []
        confidences = []
        boxes = []
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                if detection[4] > objectnessThreshold :
                    scores = detection[5:]
                    classId = np.argmax(scores)
                    confidence = scores[classId]
                    if confidence > confThreshold:
                        center_x = int(detection[0] * frameWidth)
                        center_y = int(detection[1] * frameHeight)
                        width = int(detection[2] * frameWidth)
                        height = int(detection[3] * frameHeight)
                        left = int(center_x - width / 2)
                        top = int(center_y - height / 2)
                        classIds.append(classId)
                        confidences.append(float(confidence))
                        boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            drawPred(classIds[i], confidences[i], left, top, left + width, top + height)
    def initalize_class_attributes(self):
        self.computer_vision_attributes = {
            'object_threshold': 0.5,
            'confidence_threshold': 0.5,
            'non_maximum_suppression_threshold': 0.4,
            'input_width': 416,
            'input_height': 416
        }



    def create_blob(self):
        self.initalize_class_attributes()
        blob = cv2.dnn.blobFromImage(
            frame,
            1 / 255,
            (self.computer_vision_attributes["input_width"], self.computer_vision_attributes["input_height"]),
            [0, 0, 0],
            1,
            crop=False)