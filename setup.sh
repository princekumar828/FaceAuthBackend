#!/bin/bash

# Update package list
apt-get update

# Install required system packages
apt-get install -y \
    python3-dev \
    python3-pip \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    pkg-config

# Upgrade pip
python3 -m pip install --no-cache-dir --upgrade pip

# Install base requirements first
python3 -m pip install --no-cache-dir wheel setuptools numpy==1.21.6

# Install remaining Python dependencies
python3 -m pip install --no-cache-dir -r requirements.txt
