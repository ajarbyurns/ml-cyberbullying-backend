# Use a lightweight python base image
FROM python:3.9

# Set working directory inside the container
WORKDIR /app

# Copy project files to container
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Expose the FastAPI port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]