# Use an official Python image as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install Conda
RUN apt-get update && apt-get install -y wget bzip2 && \
    wget -O miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    /opt/conda/bin/conda init && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Add Conda to PATH
ENV PATH=/opt/conda/bin:$PATH

# Copy environment.yml to the container
COPY environment.yaml .

# Create Conda environment
RUN conda env create -f environment.yaml && \
    conda clean -afy

# Activate the environment
ENV PATH=/opt/conda/envs/test/bin:$PATH

# Copy application code
COPY ["predict.py", "random_forest_model_estimators=20_max_features=1.0.bin", "./"]

# Expose the application port
EXPOSE 9696

# Set the default command to run the application
CMD ["waitress-serve", "--listen=0.0.0.0:9696", "predict:app"]