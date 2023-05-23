# Use the official TensorFlow GPU image as the base image
FROM tensorflow/tensorflow:latest-gpu

# Set the working directory
WORKDIR /home/reza/Code/dlig/

# Copy the current directory contents into the container at /app
ADD . .

# Install any necessary dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run main.py when the container launches
CMD ["python", "src/main.py"]
