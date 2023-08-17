

import cv2
import numpy as np
import math
import pyttsx3
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.camera import Camera
from kivy.clock import Clock
from kivy.graphics.texture import Texture


# Load YOLOv3 weights and configuration
yolo_net = cv2.dnn.readNet("yolov2-tiny.cfg", "yolov2-tiny.weights")

# Load COCO names (classes of objects)
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Get output layer names
output_layer_names = yolo_net.getUnconnectedOutLayersNames()

# Set minimum confidence threshold for detection
conf_threshold = 0.3


# Colors
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLACK = (0, 0, 0)

# defining the fonts
fonts = cv2.FONT_HERSHEY_COMPLEX

# Define the initial distance from the camera to the object in any suitable unit (e.g., meters)
initial_distance = 2.0  # Example: 2 meters


# Initialize the text-to-speech engine
engine = pyttsx3.init()

# ... (rest of the code)
class MainScreen(Screen):
    pass
class RealTimeObjectDetectionApp(App):
    # ... (previous code)
    def build(self):
        self.screen_manager = ScreenManager()
        self.main_screen = MainScreen(name="main")
        self.screen_manager.add_widget(self.main_screen)
        
        return self.screen_manager

    def on_start(self):
        self.layout = BoxLayout(orientation='vertical')
        self.start_button = Button(text="Start Detection", size_hint=(1, 0.1))
        self.start_button.bind(on_press=self.toggle_detection)
        self.layout.add_widget(self.start_button)
        
        self.camera = cv2.VideoCapture(0)
        self.image = Image()
        self.layout.add_widget(self.image)
        
        self.is_detecting = False
        Clock.schedule_interval(self.update, 1.0 / 30.0)
        
        self.main_screen.add_widget(self.layout)
    
    def toggle_detection(self, instance):
        self.is_detecting = not self.is_detecting
        self.start_button.text = "Stop Detection" if self.is_detecting else "Start Detection"
          
     
    def update(self, dt):
        if self.is_detecting:
            ret, frame = self.camera.read()
            if ret:
                # ... (previous code)
                # Resize image to YOLO network size (YOLOv3 uses input size of 416x416)
                height, width = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

                # Set the input to the network
                yolo_net.setInput(blob)

                # Perform object detection
                detections = yolo_net.forward(output_layer_names)


                # Process detections
                distances = []
                class_ids = []
                confidences = []
                boxes = []
                for detection in detections:
                    for obj in detection:
                        scores = obj[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]

                        # ... (previous code)
                          # Filter out weak detections (confidence threshold)
                        if confidence > conf_threshold:
                            center_x = int(obj[0] * width)
                            center_y = int(obj[1] * height)
                            w = int(obj[2] * width)
                            h = int(obj[3] * height)

                            # Get the top-left corner of the bounding box
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)

                            class_ids.append(class_id)
                            confidences.append(float(confidence))
                            boxes.append([x, y, w, h])

                            # Calculate the distance using triangulation

                            # Calculate the distance using triangulation
                            distance = initial_distance * math.sqrt(w * h / (width * height))
                            distances.append(distance)

                # ... (previous code)
                # Apply non-maximum suppression to remove overlapping boxes
                nms_indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, 0.4)

                # Draw the bounding boxes and labels on the frame
                for i in nms_indices:
                    x, y, w, h = boxes[i]
                    class_id = class_ids[i]
                    confidence = confidences[i]

                    color = (0, 255, 0)  # Green color
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    label = f"{classes[class_id]}: {confidence:.2f}, Distance: {distances[i]:.2f} meters"
                    cv2.putText(frame, label, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Convert the detected object label to speech
                    engine.say(classes[class_id])
                    engine.runAndWait()

                # ... (previous code)
                  # Convert the frame to texture and display it in Kivy Image widget
                buf1 = cv2.flip(frame, 0)
                buf = buf1.tostring()
                texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
                texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                self.image.texture = texture

    def on_stop(self):
        self.camera.release()

if __name__ == '__main__':
    RealTimeObjectDetectionApp().run()

# ... (rest of the code)
