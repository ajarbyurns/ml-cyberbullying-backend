FROM python:3.9

WORKDIR /app

# Copy only requirements first
COPY requirements.txt .

# Install dependencies (cached unless requirements.txt changes)
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]