FROM ultralytics/ultralytics:8.2.98-python
RUN pip3 install fastapi uvicorn
RUN pip3 install typing
RUN pip3 install python-multipart
RUN pip3 install aiofiles
COPY . /app
WORKDIR /app
# CMD uvicorn app:app
# Expose the port on which the application will run
# EXPOSE 8080

# Run the FastAPI application using uvicorn server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]