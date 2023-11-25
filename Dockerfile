# Use the official Python runtime as a base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /

# Copy the Flask app code into the container
#COPY . /app

# Install required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 8080

# Define environment variables


# Command to run the application
CMD ["python", "license_app.py"]
