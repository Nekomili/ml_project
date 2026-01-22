# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install Label Studio and dependencies.
RUN pip install label-studio

# Copy configuration file
COPY label_config.json /app/label_config.json
COPY src/code/create_project.py /app/create_project.py

# Expose the port Label Studio listens on.
EXPOSE 8080

# Start Label Studio
CMD ["label-studio", "start", "--port", "8080", "--host", "0.0.0.0"]