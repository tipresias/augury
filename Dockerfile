# Use an official Python runtime as a parent image
# Specifying the sha is to guarantee that CI will not try to rebuild from the
# source image (i.e. python:3.6), which apparently CIs are bad at avoiding on
# their own
FROM python:3.8.1@sha256:f0dfa61edecce35f941770264302621b1d8e25600c09dbd218798e7949083a32

# Install curl & node
RUN apt-get -y install curl \
  && curl -sL https://deb.nodesource.com/setup_8.x | bash \
  && apt-get -y install nodejs \
  && npm install -g serverless

WORKDIR /app

# Install Serverless Framework dependencies
COPY package.json package-lock.json ./
RUN npm install

# Install Python dependencies
COPY requirements.txt requirements.dev.txt /app/
RUN pip3 install --upgrade pip -r requirements.dev.txt

RUN jupyter contrib nbextension install --user \
  && jupyter nbextensions_configurator enable --user

# Add the rest of the code
COPY . /app

# Make port 8888 available for Jupyter notebooks
EXPOSE 8888

# Make port 8008 available for the app
EXPOSE 8008
