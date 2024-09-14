# Use the official Python 3.9 image as the base image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt /app/requirements.txt

# Install the required dependencies
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Set environment variables for Streamlit and other configurations
ENV PYTHONUNBUFFERED=1 \
    PATH="/home/user/.local/bin:$PATH" \
    HOME=/home/user

# Set up a new user named "user"
RUN useradd user

# Switch to the "user" user
USER user

# Set the working directory to the user's home directory
WORKDIR /home/user/app

# Copy the app content to the container and change ownership to the user
COPY --chown=user . /home/user/app

# Expose the port Streamlit will use
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "Groq_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
