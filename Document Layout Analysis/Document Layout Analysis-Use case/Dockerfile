# Use the official Python image as the base image
FROM python:3.10-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app
RUN apt-get update && apt-get install -y git && apt-get install -y g++ && apt-get install -y libgl1-mesa-dev && apt-get install -y libglib2.0-0 && apt-get install -y poppler-utils
# Install any other necessary dependencies
# RUN pip install opencv-python-headless

RUN pip install -r detectronreq.txt

RUN pip install -r layoutparserreq.txt

RUN pip install -r requirements.txt

# Define environment variable
ENV NAME World


# Expose the port your application will run on
EXPOSE 80

# Define the command to run your application
CMD ["python", "DocumentLayoutAnalysisUI.py"]
