# Specifying the sha is to guarantee that CI will not try to rebuild from the
# source image (i.e. python:3.6), which apparently CIs are bad at avoiding on
# their own.
# Using slim-buster instead of alpine based on this GH comment:
# https://github.com/docker-library/python/issues/381#issuecomment-464258800
FROM python:3.10.6-slim-buster@sha256:f17c94905c9cd56dce9ef6ce63229045a75f395f7b5b68eb69ef617079c51848

# Install linux packages
RUN apt-get --no-install-recommends update \
  # g++ is a dependency of gcc, so must come before
  && apt-get -y --no-install-recommends install g++ gcc \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Add the rest of the code
COPY . /app

CMD [ "python3", "app.py" ]
