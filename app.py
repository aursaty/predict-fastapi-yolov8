from fastapi import FastAPI, File, HTTPException
from fastapi.responses import FileResponse
from utils.utils import predict, get_image_from_bytes
import logging
from typing import Annotated
import io
import json


app = FastAPI()

logger = logging.getLogger(__name__)
logging.basicConfig(filename='myapp.log', level=logging.INFO)

@app.post('/predictClassesJson/')
async def post_predict(file: Annotated[bytes, File()]):
    """Route for prediction classes on the image with YOLO model
    
    Args:
        File: Image file

    Returns:
        JSON object with classes predicted, confidence and objects coordinates on the image 

    Raises HTTPExceptions (400, 500): Error of convertion file to the image format, other internal error 
    """
    try:
        image = get_image_from_bytes(file)
        result = predict(image)
        response = {'result': json.loads(result.to_json())}
        return response
    except IOError as e:
        raise HTTPException(status_code=400, detail="Invalid image format")
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.post('/predictClassesOnImage/')
async def post_predict(file: Annotated[bytes, File()], response_class=FileResponse):
    """Route for prediction classes on the image with YOLO model
    
    Args:
        File: Image file

    Returns:
        Users file with object with classes predicted, shown as colored boxes and text label with class and confidence on the image 

    Raises HTTPExceptions (400, 500): Error of convertion file to the image format, other internal error 
    """    
    try:
        filename = 'result.jpg'
        image = get_image_from_bytes(file)
        result = predict(image)
        image_prediction = result.save(filename)
        return FileResponse(filename)
    except IOError as e:
        raise HTTPException(status_code=400, detail="Invalid image format")
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")