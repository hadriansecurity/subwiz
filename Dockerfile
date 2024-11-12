# Start with an official Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file first (if it exists) and install dependencies
COPY requirements.txt .
copy subwiz.py .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the container
COPY . .


# Set the default command to run the application
CMD ["python", "subwiz.py"]
