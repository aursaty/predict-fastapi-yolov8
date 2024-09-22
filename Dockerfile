FROM ultralytics/ultralytics:8.2.98-python
RUN pip3 install fastapi uvicorn
COPY . /app
WORKDIR /app
CMD python main.py
