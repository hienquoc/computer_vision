import cv2
import numpy as np

import utility.utility
from utility.utility import Utility
import os
import sys
import os.path
utility.utility.utility_debug_status = False

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
            elif key == 'initialize_class_attributes':
                self.initalize_class_attributes()
            elif key == 'load_model_and_classes':
                self.load_model_and_classes()
            elif key == 'yolo_input_network':
                self.yolo_input_network()

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


    def read_frame_by_dictionary(self):
        # Read first frame
        self.method_selector_dictionary = {
            'initialize_class_attributes': 'initialize_class_attributes',
            'load_model_and_classes': 'load_model_and_classes',
        }
        self.method_selector()
        while True:
            #self.video_status, self.video_frame = self.video.read()

            self.computer_vision_attributes["video_status"], self.computer_vision_attributes["video_frame"] = self.video.read()
            '''
            # Scale image down
            self.computer_vision_attributes["video_frame"] = cv2.resize(self.computer_vision_attributes["video_frame"],
                                                                        None,
                                                                        fx=self.computer_vision_attributes["input_width"] / self.computer_vision_attributes["video_frame"].shape[1],
                                                                        fy=self.computer_vision_attributes["input_height"] / self.computer_vision_attributes["video_frame"].shape[0],
                                                                        interpolation=cv2.INTER_LINEAR)
            '''
            self.debug.title = 'class: Computer_vision def: read_frame_by_dictionary'
            self.debug.debug_variable_dictionary = {'self.computer_vision_attributes["video_status"]': self.computer_vision_attributes["video_status"],
                                                    'self.computer_vision_attributes["video_frame"]': self.computer_vision_attributes["video_frame"]}
            #self.debug.print_value_dictionary()
            self.method_selector_dictionary = {
                'yolo_input_network': 'yolo_input_network'
            }
            self.method_selector()
            cv2.imshow('Video', self.computer_vision_attributes["video_frame"])
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()



    def getOutputsNames(self, net):
        # Get the names of all the layers in the network

        layer_names = net.getLayerNames()
        print(layer_names)
        print(net.getUnconnectedOutLayers())
        result = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        self.debug.title = 'class: Computer_vision def: getOutputsNames'
        self.debug.debug_variable_dictionary = {'result': result}
        self.debug.print_value_dictionary()
        return result

# Draw the predicted bounding box
    def drawPred(self, classId, conf, left, top, right, bottom):
        self.debug.title = 'class: Computer_vision def: drawPred - Start'
        self.debug.debug_variable_dictionary = {'classId': classId,
                                                'conf': conf,
                                                'left': left,
                                                'top': top,
                                                'right': right,
                                                'bottom': bottom}
        self.debug.print_value_dictionary()
        # Draw a bounding box.
        cv2.rectangle(self.computer_vision_attributes["video_frame"], (left, top), (right, bottom), (255, 178, 50), 3)

        label = '%.2f' % conf

        # Get the label for the class name and its confidence
        if self.computer_vision_attributes['class']:
            assert (classId < len(self.computer_vision_attributes['class']))
            label = '%s:%s' % (self.computer_vision_attributes['class'][classId], label)

        # Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv2.rectangle(self.computer_vision_attributes["video_frame"], (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
                      (255, 255, 255), cv2.FILLED)
        cv2.putText(self.computer_vision_attributes["video_frame"], label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)
        self.debug.title = 'class: Computer_vision def: drawPred - End'
        self.debug.debug_variable_dictionary = {'classId': classId,
                                                'conf': conf,
                                                'left': left,
                                                'top': top,
                                                'right': right,
                                                'bottom': bottom,
                                                "self.computer_vision_attributes['class']": self.computer_vision_attributes['class'],
                                                'labelSize': labelSize,
                                                'baseLine': baseLine

                                                }
        self.debug.print_value_dictionary()

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
        self.debug.title = 'class: Computer_vision def: postprocess - Start'
        self.debug.debug_variable_dictionary = {'frameHeight': frameHeight,
                                                'frameWidth': frameWidth,
                                                'outs[0:5]': outs[0:5]}
        #self.debug.print_value_dictionary()
        for out in outs:
            out_4_greaterthan_object_threshold = out[4] > self.computer_vision_attributes['object_threshold']
            self.debug.title = f'class: Computer_vision def: postprocess - out loop = {out}'
            self.debug.debug_variable_dictionary = {'frameHeight': frameHeight,
                                                    'frameWidth': frameWidth,
                                                    'detection_4_greaterthan_object_threshold':out_4_greaterthan_object_threshold}
            #self.debug.print_value_dictionary()
            if out_4_greaterthan_object_threshold:
                scores = out[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                confidence_greaterthan_confidence_threshold = confidence > self.computer_vision_attributes['confidence_threshold']
                self.debug.debug_variable_dictionary = {'frameHeight': frameHeight,
                                                        'frameWidth': frameWidth,
                                                        'out_4_greaterthan_object_threshold': out_4_greaterthan_object_threshold,
                                                        'scores': scores,
                                                        'classId': classId,
                                                        'confidence': confidence,
                                                        'confidence_greaterthan_confidence_threshold': confidence_greaterthan_confidence_threshold}
                self.debug.print_value_dictionary()
                if confidence_greaterthan_confidence_threshold:
                    center_x = int(out[0] * frameWidth)
                    center_y = int(out[1] * frameHeight)
                    width = int(out[2] * frameWidth)
                    height = int(out[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
                    self.debug.debug_variable_dictionary = {'center_x': center_x,
                                                            'center_y': center_y,
                                                            'width': width,
                                                            'height': height,
                                                            'left': left,
                                                            'top': top,
                                                            'classIds': classIds,
                                                            'confidences': confidences,
                                                            'boxes': boxes}
                    self.debug.print_value_dictionary()


        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes,
                                   confidences,
                                   self.computer_vision_attributes['confidence_threshold'],
                                   self.computer_vision_attributes['non_maximum_suppression_threshold'])
        self.debug.title = 'class: Computer_vision def: postprocess - Start Indices Loop'
        self.debug.debug_variable_dictionary = {'indices': indices,
                                                'boxes': boxes,
                                                'confidences': confidences,
                                                "self.computer_vision_attributes['confidence_threshold']": self.computer_vision_attributes['confidence_threshold'],
                                                "self.computer_vision_attributes['non_maximum_suppression_threshold']": self.computer_vision_attributes['non_maximum_suppression_threshold']
                                                }
        self.debug.print_value_dictionary()
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            self.drawPred(classIds[i], confidences[i], left, top, left + width, top + height)
            self.debug.title = 'class: Computer_vision def: postprocess - End'
            self.debug.debug_variable_dictionary = {'i': i,
                                                    'box': box,
                                                    'left': left,
                                                    'top': top,
                                                    'width': width,
                                                    'top': top,
                                                    'height': height,
                                                    'drawPred': self.drawPred,
                                                    }
            self.debug.print_value_dictionary()

    def initalize_class_attributes(self):
        self.computer_vision_attributes.update({
            'object_threshold': 0.5,
            'confidence_threshold': 0.5,
            'non_maximum_suppression_threshold': 0.4,
            'input_width': 416,
            'input_height': 416
        })

    def create_blob(self):

        blob = cv2.dnn.blobFromImage(
            self.computer_vision_attributes["video_frame"],
            1 / 255,
            (self.computer_vision_attributes["input_width"], self.computer_vision_attributes["input_height"]),
            [0, 0, 0],
            1,
            crop=False)
        self.debug.title = 'class: Computer_vision def: create_blob'
        self.debug.debug_variable_dictionary = {'create_blob': blob}
        #self.debug.print_value_dictionary()
        return blob

    def yolo_input_network(self):
        # Set the blob as input to the network
        self.initalize_class_attributes()
        self.computer_vision_attributes['network'].setInput(self.create_blob())   # Set the input blob
        self.debug.title = 'class: Computer_vision def: yolo_input_network - Start'
        self.debug.debug_variable_dictionary = self.computer_vision_attributes
        #self.debug.print_value_dictionary()
        # Run the forward pass to get output from the output layers
        self.computer_vision_attributes['output'] = self.computer_vision_attributes['network'].forward(self.getOutputsNames(self.computer_vision_attributes['network']))
        # Remove the bounding boxes with low confidence
        self.postprocess(self.computer_vision_attributes['video_frame'], self.computer_vision_attributes['output'])
        # Put efficiency information
        timing_for_each_layer, _ = self.computer_vision_attributes['network'].getPerfProfile()
        label = 'Inference time: %.2f ms' % (timing_for_each_layer * 1000.0 / cv2.getTickFrequency())
        cv2.putText(self.computer_vision_attributes['video_frame'], label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        # Display the frame
        print(label)
        self.debug.title = 'class: Computer_vision def: yolo_input_network - End'
        self.debug.debug_variable_dictionary = {
            'self.computer_vision_attributes["network"].setInput(self.create_blob())  ': self.computer_vision_attributes["network"].setInput(self.create_blob()),
            'self.computer_vision_attributes["output"]': self.computer_vision_attributes["output"],
            'timing_for_each_layer': timing_for_each_layer
            }
        #self.debug.print_value_dictionary()

    def load_model_and_classes(self):
        self.computer_vision_attributes['network'] = cv2.dnn.readNetFromDarknet(self.computer_vision_attributes['local_file_path'] + "\yolov3.cfg",
                                                                                self.computer_vision_attributes['local_file_path'] + "\yolov3.weights"

                                                                                )
        self.computer_vision_attributes['class'] = None
        with open(self.computer_vision_attributes['local_file_path'] + "\coco.names", 'rt') as f:
            self.computer_vision_attributes['class'] = f.read().rstrip('\n').split('\n')
        self.debug.title = 'class: Computer_vision def: load_model_and_classes'
        self.debug.debug_variable_dictionary = self.computer_vision_attributes
        self.debug.print_value_dictionary()

    def yolo_object_dector_in_video(self):
        computer_vision_class_file_path = os.path.abspath(os.getcwd())
        self.read_video()
        while True:
            self.computer_vision_attributes["video_status"], self.computer_vision_attributes["video_frame"] = self.video.read()
            if self.computer_vision_attributes["video_status"] == False:
                break
            self.yolo_object_detector(self.computer_vision_attributes["video_frame"])
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break








    def yolo_object_detector(self, image):
        computer_vision_class_file_path = os.path.abspath(os.getcwd())
        self.computer_vision_attributes = {'local_file_path': computer_vision_class_file_path}
        # Load Yolo
        net = cv2.dnn.readNet(self.computer_vision_attributes['local_file_path'] + "\yolov3.weights",
                              self.computer_vision_attributes['local_file_path'] + "\yolov3.cfg")
        classes = []
        with open(self.computer_vision_attributes['local_file_path'] + "\coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
        layer_names = net.getLayerNames()

        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        colors = np.random.uniform(0, 255, size=(len(classes), 3))

        # Loading image
        img = image
        img = cv2.resize(img, None, fx=0.4, fy=0.4)
        height, width, channels = img.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                if label != '':
                    color = colors[class_ids[i]]
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(img, label, (x, y + 30), font, 3, (255, 0, 0), 3)

        cv2.imshow("Image", img)



    def object_tracker_main(self):
        # Initialize tracker
        computer_vision_class_file_path = os.path.abspath(os.getcwd())
        self.computer_vision_attributes = {'local_file_path': computer_vision_class_file_path}
        self.video_file_path = computer_vision_class_file_path + '\soccer-ball.mp4'
        self.setup_tracker()
        self.read_video()
        self.read_frame_by_dictionary()