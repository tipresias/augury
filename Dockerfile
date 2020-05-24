# Use an official Python runtime as a parent image
# Specifying the sha is to guarantee that CI will not try to rebuild from the
# source image (i.e. python:3.6), which apparently CIs are bad at avoiding on
# their own.
# Using slim-buster instead of alpine based on this GH comment:
# https://github.com/docker-library/python/issues/381#issuecomment-464258800
FROM python:3.7.5-slim-buster@sha256:59af1bb7fb92ff97c9a23abae23f6beda13a95dbfd8100c7a2f71d150c0dc6e5

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

CMD [ "python3", "app.py" ]
