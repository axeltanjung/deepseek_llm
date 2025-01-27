# Use the official Python 3.8 image as a base image
FROM python:3.8-slim

# Set the working directory inside the container
WORKDIR /app

# Copy your local project files into the container (including your script and requirements file)
COPY . /app

# Install necessary system dependencies (you might need some libraries for GPU support in PyTorch)
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libomp-dev \
    libopenblas-dev

# Install the required Python packages
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Set the entrypoint to run your script
CMD ["python", "./src/main.py"]
