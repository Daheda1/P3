import os
import cv2
import torch
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, Response, render_template_string
from ultralytics import YOLO

from modelparts.modelStructure import UNet
from modelparts.imagePreprocessing import scale_image, pad_image_to_target

# Initialize Flask app
app = Flask(__name__)

def preprocess_frame(frame, target_size, device):
    """
    Preprocesses the input frame for the UNet model.
    """
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(frame_rgb)
    
    # Scale and pad
    scaled_image, scale_ratio = scale_image(pil_image, target_size, divideable_by=32)
    #print(f"Scaled image size: {scaled_image.size}, Scale ratio: {scale_ratio}")
    
    padded_image, padding = pad_image_to_target(scaled_image, target_size)
    #print(f"Padded image size: {padded_image.size}, Padding: {padding}")
    
    # Convert back to NumPy
    padded_np = np.array(padded_image).astype(np.float32) / 255.0  # Normalize
    
    # Convert to tensor
    frame_tensor = torch.from_numpy(padded_np).permute(2, 0, 1).unsqueeze(0).to(device)  # (1, C, H, W)
    
    return frame_tensor

def draw_bounding_boxes(frame, boxes, class_names):
    """
    Draws bounding boxes on the frame.
    """
    # Ensure frame is a contiguous array
    frame = np.ascontiguousarray(frame)

    # Validate frame
    if not (isinstance(frame, np.ndarray) and frame.dtype == np.uint8 and frame.ndim == 3 and frame.shape[2] == 3):
        raise ValueError("Frame must be a numpy array with shape (H, W, 3) and dtype uint8")
    
    # Debugging statements
    #print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}, contiguous: {frame.flags['C_CONTIGUOUS']}")

    for box in boxes:
        xyxy = box.xyxy  # Tensor of shape (N, 4)
        conf = box.conf  # Tensor of shape (N,)
        cls = box.cls    # Tensor of shape (N,)

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i].tolist()
            confidence = conf[i].item()
            class_id = int(cls[i].item())
            class_name = class_names.get(class_id, "Unknown")

            # Debug: Print bounding box details
            #print(f"Drawing box: {class_name} with confidence {confidence:.2f} at [{x1}, {y1}, {x2}, {y2}]")

            # Define box color and thickness
            color = (0, 255, 0)  # Green color for bounding boxes
            thickness = 2

            # Draw the rectangle
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

            # Put the label near the bounding box
            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(int(y1) - label_size[1] - 10, 0)
            cv2.rectangle(frame, 
                          (int(x1), label_ymin),
                          (int(x1) + label_size[0], label_ymin + label_size[1] + 10),
                          color, -1)
            cv2.putText(frame, label, (int(x1), label_ymin + label_size[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

def generate_frames():
    """
    Generator function that yields video frames in JPEG format.
    """
    # Configuration
    class_ids = [1, 8, 39, 5, 2, 15, 56, 41, 16, 3, 0, 60]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    #print(f"Using device: {device}")
    
    checkpoint_path = "checkpoint_epoch_7.pth"
    target_size = (640, 640)  # (width, height)

    # Verify checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

    # Initialize and load UNet model
    unet_model = UNet(in_channels=3, out_channels=3).to(device)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            unet_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            unet_model.load_state_dict(checkpoint)
    except RuntimeError as e:
        print(f"Error loading checkpoint: {e}")
        return
    unet_model.eval()

    # Initialize YOLO model
    yolo_model = YOLO("yolo11n.pt")
    yolo_model.to(device)
    yolo_model.classes = class_ids  # Filter classes

    # Open webcam
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from webcam.")
                break

            # Preprocess frame
            input_tensor = preprocess_frame(frame, target_size, device)

            # UNet Inference
            enhanced_output = unet_model(input_tensor)

            # Postprocess UNet output
            enhanced_frame = enhanced_output.squeeze().permute(1, 2, 0).cpu().numpy()
            enhanced_frame = np.clip(enhanced_frame * 255.0, 0, 255).astype(np.uint8)

            # YOLO Inference (YOLO expects RGB)
            yolo_results = yolo_model(enhanced_frame)

            # Draw bounding boxes on the enhanced frame
            for result in yolo_results:
                boxes = result.boxes
                class_names = result.names

                if boxes is None or len(boxes) == 0:
                    continue

                draw_bounding_boxes(enhanced_frame, boxes, class_names)

            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', enhanced_frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()

            # Yield frame in byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """
    Home page that displays the video feed.
    """
    # Simple HTML template to display the video stream
    return render_template_string('''
        <html>
            <head>
                <title>Surveillance Video Feed</title>
            </head>
            <body>
                <h1>Live Surveillance Feed</h1>
                <img src="{{ url_for('video_feed') }}" width="800" />
            </body>
        </html>
    ''')

@app.route('/video_feed')
def video_feed():
    """
    Video streaming route. Put this in the src attribute of an img tag.
    """
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def run_flask_app():
    """
    Runs the Flask app.
    """
    app.run(host='0.0.0.0', port=5001, threaded=True)

if __name__ == "__main__":
    run_flask_app()