import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detection App")
        self.root.geometry("800x600")  # Set window size
        self.root.configure(bg="#282c34")  # Set background color

        # Create a label to display the video feed
        self.video_label = tk.Label(root, bg="black")
        self.video_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Create buttons with enhanced styles
        self.open_camera_btn = tk.Button(root, text="Open Camera", command=self.open_camera,
                                         bg="#61afef", fg="white", font=("Arial", 12, "bold"))
        self.open_camera_btn.pack(side=tk.LEFT, padx=10, pady=10)

        self.close_camera_btn = tk.Button(root, text="Close Camera", command=self.close_camera,
                                          bg="#e06c75", fg="white", font=("Arial", 12, "bold"))
        self.close_camera_btn.pack(side=tk.LEFT, padx=10, pady=10)

        self.quit_btn = tk.Button(root, text="Quit", command=self.quit_app,
                                  bg="#98c379", fg="white", font=("Arial", 12, "bold"))
        self.quit_btn.pack(side=tk.RIGHT, padx=10, pady=10)

        self.cap = None
        self.running = False

        # Load YOLO model
        self.net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        self.layer_names = self.net.getLayerNames()
        try:
            self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        except IndexError:
            self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

        # Load the COCO class labels
        with open("coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

    def open_camera(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open the webcam.")
                return

            self.running = True
            self.update_frame()

    def update_frame(self):
        if self.running:
            ret, frame = self.cap.read()
            height, width, channels = frame.shape

            # Detecting objects
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.net.setInput(blob)
            outs = self.net.forward(self.output_layers)

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

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(self.classes[class_ids[i]])
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Convert the frame to a format suitable for Tkinter
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            self.video_label.after(10, self.update_frame)

    def close_camera(self):
        if self.running:
            self.running = False
            self.cap.release()
            self.video_label.config(image='')

    def quit_app(self):
        self.close_camera()
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()