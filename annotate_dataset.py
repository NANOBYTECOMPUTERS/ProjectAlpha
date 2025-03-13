import os
from ultralytics import YOLO
from config import cfg

# Define base directory
base_dir = cfg.train_base_dir

# Load pre-trained model
model = YOLO(f"models/{cfg.annotate_model_name}")
print(model.names)

# Define subdirectories to process
subdirs = ['train', 'valid', 'test']

# Process each subdirectory
for subdir in subdirs:
    # Set input and output directories
    image_dir = os.path.join(base_dir, subdir, 'images')
    label_dir = os.path.join(base_dir, subdir, 'labels')
    
    # Create labels directory if it doesn't exist
    os.makedirs(label_dir, exist_ok=True)
    
    # Process each image in the images directory
    for image_file in os.listdir(image_dir):
        if image_file.lower().endswith(('.jpg', '.png')):
            # Full path to input image
            image_path = os.path.join(image_dir, image_file)
            
            # Run model prediction
            results = model(image_path)
            result = results[0]
            
            # Get image dimensions
            width, height = result.orig_shape
            
            # Create annotation file path (in labels directory)
            annotation_file = os.path.join(label_dir, 
                                        os.path.splitext(image_file)[0] + '.txt')
            
            # Write predictions to annotation file
            with open(annotation_file, 'w') as f:
                for box in result.boxes:
                    class_id = int(box.cls)
                    x1, y1, x2, y2 = box.xyxy[0]
                    # Convert to YOLO format (normalized coordinates)
                    x_center = (x1 + x2) / 2 / width
                    y_center = (y1 + y2) / 2 / height
                    box_width = (x2 - x1) / width
                    box_height = (y2 - y1) / height
                    f.write(f"{class_id} {x_center} {y_center} {box_width} {box_height}\n")

    print(f"Finished processing {subdir} dataset")