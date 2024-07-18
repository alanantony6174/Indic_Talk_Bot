# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Update package lists and install necessary tools
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    libsndfile1-dev \
    ffmpeg \
    python3-enchant \
    mecab \
    libmecab-dev \
    mecab-ipadic-utf8 \
    libatlas-base-dev \
    liblapack-dev \
    gfortran \
    libfreetype6-dev \
    libpng-dev \
    libpq-dev \
    libxml2-dev \
    libxslt1-dev \
    libldap2-dev \
    libsasl2-dev \
    libopenblas-dev \
    libblas-dev \
    liblapack-dev \
    git \
    curl \
    snapd \
    portaudio19-dev \
    && apt-get clean \
    pulseaudio \
    pulseaudio-utils \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as the default python3 version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install pip upgrade and setuptools
RUN pip3 install -U pip setuptools wheel

# Install sentencepiece from GitHub
RUN pip3 install git+https://github.com/huggingface/transformers.git sentencepiece

# Install PyTorch and related packages
RUN pip3 install -U torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY . /app

# Install Python packages from requirements.txt
RUN pip3 install -r requirements.txt

# Copy Trainer directory and install its dependencies
RUN pip3 install -e /app/Trainer[all]

# Copy TTS directory and install its dependencies
RUN pip3 install -e /app/TTS[all]


# Expose the port
EXPOSE 7777

# Start both ollama server and your application
CMD ["streamlit", "run", "app.py"]