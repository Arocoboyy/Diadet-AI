//requirements.txt 
Flask
numpy
inference_sdk

//Dockerfile
# Use an official Python runtime as a parent image
FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]
