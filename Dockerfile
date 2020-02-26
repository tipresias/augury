# Use an official Python runtime as a parent image
# Specifying the sha is to guarantee that CI will not try to rebuild from the
# source image (i.e. python:3.6), which apparently CIs are bad at avoiding on
# their own.
# Using slim-buster instead of alpine based on this GH comment:
# https://github.com/docker-library/python/issues/381#issuecomment-464258800
FROM python:3.8.2-slim-buster@sha256:a11a920a223bd9cb3860f6ee879d75089a49a1b3ddf77dd9cb93d710f5d8d96b

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
  && jupyter nbextensions_configurator enable --user \
  && jupyter nbextension enable --py widgetsnbextension \
  && jupyter nbextension enable codefolding/main \
  && jupyter nbextension enable execute_time/ExecuteTime \
  && jupyter nbextension enable toc2/main \
  && jupyter nbextension enable collapsible_headings/main \
  && jupyter nbextension enable notify/notify \
  && jupyter nbextension enable codefolding/edit

# Add the rest of the code
COPY . /app

# Make port 8888 available for Jupyter notebooks
EXPOSE 8888

# Make port 8008 available for the app
EXPOSE 8008
