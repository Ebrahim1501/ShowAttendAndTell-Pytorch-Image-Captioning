# Use Python 3.12 base image
FROM python:3.12-slim

# Install system dependencies for OpenCV and other libraries
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio && \
    pip install --no-cache-dir -r requirements.txt

# Expose port for Streamlit
EXPOSE 8501

# Copy the rest of your application into the container
COPY . .

# Command to run the application
CMD ["streamlit", "run", "App.py"]
