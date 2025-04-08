# Use an official lightweight Python image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install scikit-learn==1.4.1.post1


# Copy application files
COPY app.py score.py logistic_model.pkl ./

# Expose port 5000 for Flask app
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
