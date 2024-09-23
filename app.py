from fastapi import FastAPI, File, HTTPException
from fastapi.responses import FileResponse
from utils.utils import predict, get_image_from_bytes
import logging
from typing import Annotated
import io
import json
# import aiofiles 


app = FastAPI()


@app.post('/predictClassesJson/')
async def post_predict(file: Annotated[bytes, File()]):
    try:
        image = get_image_from_bytes(file)
    except IOError:
        raise HTTPException(status_code=400, detail="File wasn't reconized as image! Please try again and send an image file.")
    result = predict(image)
    image_prediction = result.save('result.jpg')
    # result.save(image_output, format='JPG')
    response = {
        'result': json.loads(result.to_json())
    }
    return response


@app.post('/predictClassesOnImage/')
async def post_predict(file: Annotated[bytes, File()], response_class=FileResponse):

    filename = 'result.jpg'
    image = get_image_from_bytes(file)
    result = predict(image)
    image_prediction = result.save(filename)
    return FileResponse(filename)
