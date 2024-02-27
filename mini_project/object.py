import cv2
import numpy as np
from openvino.inference_engine import IECore
import ov


# Define the function to load the model
def load_model(model_name, precision):
    # Initialize the IECore object
    ie = IECore()

    # Load the IR files
    model_xml = "/home/busan/openvino_notebooks/mini-pro/model/ssdlite_mobilenet_v2_fp16.xml"
    model_bin = "/home/busan/openvino_notebooks/mini-pro/model/ssdlite_mobilenet_v2_fp16.bin"
    net = ie.read_network(model=model_xml, weights=model_bin)

    # Set the device
    device = "CPU"

    # Load the model to the device
    compiled_model = ie.load_network(network=net, device_name=device, num_requests=1)

    # Get the input and output layers
    input_layer = net.input_info[next(iter(net.input_info))]
    output_layer = next(iter(net.outputs))

    return compiled_model, input_layer, output_layer

# Define the function to process the results
def process_results(results, input_shape, output_shape, conf_thresh=0.5):
    # Extract the scores, boxes, and labels from the results
    scores = results[output_shape[2]]
    boxes = results[output_shape[1]]
    labels = results[output_shape[3]]

    # Filter out the low-confidence detections
    keep = np.where((scores >= conf_thresh).squeeze())[0]
    scores = scores[keep]
    boxes = boxes[0, keep]
    labels = labels[keep]

    # Convert the boxes from normalized coordinates to pixel coordinates
    height, width, _ = input_shape
    boxes = boxes * np.array([width, height, width, height])

    return boxes, labels, scores

# Define the function to draw the bounding boxes and labels on the frame
def draw_boxes(frame, boxes, labels, classes):
    for box, label, score in zip(boxes, labels, scores):
        # Convert the box coordinates from (x1, y1, x2, y2) to (x1, y1, w, h)
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1

        # Draw the bounding box on the frame
        cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)

        # Draw the label and score on the frame
        label_text = f"{classes[label]}: {score:.2f}"
        cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Define the function to run the object detection
def run_object_detection(model, input_layer, output_layer, device="CPU"):
    # Initialize the OpenVINO runtime core
    core = ov.Core()

    # Load the model to the device
    compiled_model = core.compile_model(model=model, device_name=device)

    # Create a video capture object
    cap = cv2.VideoCapture(0)

    # Get the input and output shapes
    input_shape = input_layer.shape
    output_shape = output_layer.shape

    # Initialize the FPS counter
    fps = cv2.getTickFrequency() / 1000
    prev_time = cv2.getTickCount()

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        if not ret:
            break

        # Resize the frame to the input shape of the model
        frame = cv2.resize(frame, (input_shape[3], input_shape[2]))

        # Convert the frame to a batch of images
        image_batch = np.expand_dims(frame, axis=0)

        # Run the inference
        start_time = cv2.getTickCount()
        results = compiled_model([image_batch])[output_layer]
        end_time = cv2.getTickCount()

        # Post-process the results
        boxes, labels, scores = process_results(results, input_shape, output_shape)

        # Draw the bounding boxes and labels on the frame
        draw_boxes(frame, boxes, labels, classes)

        # Calculate the FPS
        curr_time = cv2.getTickCount()
        fps_value = 1000 / (curr_time - prev_time)
        prev_time = curr_time

        # Display the FPS and the frame
        cv2.putText(frame, f"FPS: {int(fps_value)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Object Detection", frame)

        # Exit the loop if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

# Define the function to load the classes
def load_classes(class_file):
    with open(class_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names
# Load the model, input layer, and output layer
model, input_layer, output_layer = load_model(model_name="ssdlite_mobilenet_v2_fp16", precision="FP16")
# Load the classes
classes = load_classes("/home/busan/openvino_notebooks/mini-pro/object_labels.txt")

# Run the object detection
run_object_detection(model, input_layer, output_layer)