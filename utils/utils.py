from ultralytics import YOLO
from PIL import Image
import pydantic
import io

# Load a COCO-pretrained YOLOv8n model
model = YOLO("yolov8n.pt")


def get_image_from_bytes(binary_image: bytes) -> Image:
    """Safely convert image from bytes to PIL RGB format
    
    Args:
        binary_image (bytes): The binary representation of the image
    
    Returns:
        PIL.Image: The image in PIL RGB format
    """
    try:
        input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
        return input_image
    except (IOError, SyntaxError):
        raise IOError("Image expected, check file extention!")
        return False


def predict(file: bytes):
    print('processing started')
    result = model(file)[0]

    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    # result.save(filename="result.jpg") 
    return result
