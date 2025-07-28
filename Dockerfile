# Dockerfile

FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app

# Install system dependencies required by PyMuPDF (fitz)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the pre-trained model and both Python scripts
COPY submission_model_definitive.txt .
COPY outline_extractor.py .
COPY process_pdfs.py .

# Command to run when the container starts
CMD ["python", "process_pdfs.py"]