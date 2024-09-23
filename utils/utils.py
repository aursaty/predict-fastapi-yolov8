from ultralytics import YOLO
from PIL import Image
import pydantic
import io
import logging
from config import YOLO_MODEL_USED, LOG_FILEPATH

logger = logging.getLogger(__name__)
logging.basicConfig(filename='myapp.log', level=logging.INFO)
# Load a COCO-pretrained YOLOv8n model
model = YOLO(YOLO_MODEL_USED)
logger.info(f"YOLO Model ({YOLO_MODEL_USED}) initiated")


def get_image_from_bytes(binary_image: bytes) -> Image:
    """Safely convert image from bytes to PIL RGB format
    
    Args:
        binary_image (bytes): The binary representation of the image
    
    Returns:
        PIL.Image: The image in PIL RGB format

    Raises IOError: Error of convertion file to the image format
    """
    try:
        logger.info("@get_image_from_bytes called")
        input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
        return input_image
    except (IOError, SyntaxError) as e:
        logger.error("PIL IOError: Image reading error.")
        raise IOError
    

def predict(file: bytes):
    """ Predict object on the image relates to specified classes

    Args:
        PIL.Image: The image in PIL RGB format

    Returns:
        Response from ultralytics.YOLO class  
    """
    logger.info("@predict called")
    result = model(file)[0]

    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    return result
