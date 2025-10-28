# 1. Start from an official Python base image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the dependency list
COPY requirements.txt requirements.txt

# 4. Install all the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your application code
# (src folder, models folder, config.yaml)
COPY src/ src/
COPY models/ models/
COPY config.yaml config.yaml

# 6. Expose the port your app runs on
EXPOSE 8000

# 7. Define the command to run your app
# This will run: uvicorn src.main:app --host 0.0.0.0 --port 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]