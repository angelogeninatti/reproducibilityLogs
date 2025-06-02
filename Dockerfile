FROM python:3.12

# Copy requirements and install dependencies first
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Copy your application code to a specific path
COPY . /app/

# Copy the wrapper script
COPY run_analysis.sh /run_analysis.sh
RUN chmod +x /run_analysis.sh

# Set working directory
WORKDIR /app

# Use the wrapper script instead of direct python call
CMD ["/run_analysis.sh"]