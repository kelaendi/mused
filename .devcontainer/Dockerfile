# Use a Python base image
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    default-jre \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /workspace

# Copy the folder contents into the container
COPY . /workspace

# List the contents of /workspace to verify the directory structure after the copy
# RUN ls -al /workspace