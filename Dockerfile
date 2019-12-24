# Use an official Python runtime as a parent image
# Specifying the sha is to guarantee that CI will not try to rebuild from the
# source image (i.e. python:3.6), which apparently CIs are bad at avoiding on
# their own.
# Using slim-buster instead of alpine based on this GH comment:
# https://github.com/docker-library/python/issues/381#issuecomment-464258800
FROM python:3.8.1-slim-buster@sha256:dc9c4de1bb38720f70af28e8071f324052725ba122878fbac784be9b03f41590

# Install linux packages & node
RUN apt-get --no-install-recommends update \
  # g++ is a dependency of gcc, so must come before
  && apt-get -y --no-install-recommends install g++ gcc curl \
  && curl -sL https://deb.nodesource.com/setup_12.x | bash - \
  && apt-get --no-install-recommends install -y nodejs \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Serverless Framework dependencies
COPY package.json package-lock.json ./
RUN npm install

# Install Python dependencies
COPY requirements.txt requirements.dev.txt ./
RUN pip3 install -r requirements.dev.txt \
  && jupyter contrib nbextension install --user \
  && jupyter nbextensions_configurator enable --user

# Add the rest of the code
COPY . /app

# Make port 8888 available for Jupyter notebooks
EXPOSE 8888

# Make port 8008 available for the app
EXPOSE 8008
