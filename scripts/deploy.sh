#!/bin/bash

set -euo pipefail

PORT=8008

gcloud builds submit --config cloudbuild.yaml --quiet

GOOGLE_ENV_VARS="
AFL_DATA_SERVICE=${AFL_DATA_SERVICE},\
AFL_DATA_SERVICE_TOKEN=${AFL_DATA_SERVICE_TOKEN},\
DATA_SCIENCE_SERVICE_TOKEN=${DATA_SCIENCE_SERVICE_TOKEN},\
PYTHON_ENV=production,\
ROLLBAR_TOKEN=${ROLLBAR_TOKEN},\
TIPRESIAS_APP_TOKEN=${TIPRESIAS_APP_TOKEN},\
GIT_PYTHON_REFRESH=quiet
"

gcloud run deploy augury \
  --quiet \
  --image gcr.io/${PROJECT_ID}/augury \
  --memory 4Gi \
  --timeout 900 \
  --region australia-southeast1 \
  --max-instances 5 \
  --concurrency 1 \
  --platform managed \
  --update-env-vars ${GOOGLE_ENV_VARS}
