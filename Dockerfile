FROM python:3.11

# Set the working directory to /app
WORKDIR /app

# Install pre-requisites
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install any needed packages using pip
COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY src/* /app
COPY yolov8 /app/yolov8

VOLUME /app/runs

VOLUME /app/datasets

ENV CUBLAS_WORKSPACE_CONFIG=:4096:8

# Run classification_runner.py when the container launches
CMD ["python", "classification_runner.py"]
