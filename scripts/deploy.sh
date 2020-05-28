#!/bin/bash

set -euo pipefail

# We need travis_wait for Travis CI builds, because installing R packages takes forever.
# On a local machine, we can just build & deploy as normal.
if [ -n "$(LC_ALL=C type -t travis_wait)" ] && [ "$(LC_ALL=C type -t travis_wait)" = function ]
then
  travis_wait 30 gcloud builds submit --config cloudbuild.yaml
else
  gcloud builds submit --config cloudbuild.yaml
fi

gcloud beta run deploy augury \
  --image gcr.io/${PROJECT_ID}/augury \
  --memory 2048Mi \
  --region us-central1 \
  --platform managed \
  --update-env-vars AFL_DATA_SERVICE_TOKEN=${AFL_DATA_SERVICE_TOKEN},PYTHON_ENV=production,DATA_SCIENCE_SERVICE_TOKEN=${DATA_SCIENCE_SERVICE_TOKEN},AFL_DATA_SERVICE=${AFL_DATA_SERVICE}
