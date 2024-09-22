from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from utils.utils import predict, get_image_from_bytes
import logging
from typing import Annotated
import io
import json


app = FastAPI(
    title='YOLOv8 - Predict Objects on pic'
)


@app.post('/predict/')
async def post_predict(file: Annotated[bytes, File()]):
    logging.info(123)
    image = get_image_from_bytes(file)
    result = predict(image)
    image_prediction = result.save('result.jpg')
    responces = {
        200: {
            'result': json.loads(result.to_json()),
            'file': FileResponse(image_prediction)
        }
    }
    # logging.info(len(result))
    return responces
