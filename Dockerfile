FROM python:3.11

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY src/* /app
COPY yolov8 /app/yolov8


# Install pre-requisites
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y


# Install any needed packages using pip
RUN pip install ultralytics

VOLUME /app/runs

VOLUME /app/datasets

# Run classification_runner.py when the container launches
CMD ["python", "classification_runner.py"]
