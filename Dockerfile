# Use a lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install required system tools
RUN apt-get update && apt-get install -y unzip && apt-get clean

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the app
COPY . .

# Expose FastAPI port
ENV PORT=7860

# Run FastAPI app
CMD ["uvicorn", "backend.wise_well_app:app", "--host", "0.0.0.0", "--port", "7860"]
