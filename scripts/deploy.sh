#!/bin/bash

set -euo pipefail

docker run \
  -e PROJECT_ID=${PROJECT_ID} \
  -e AFL_DATA_SERVICE=${AFL_DATA_SERVICE} \
  -e GOOGLE_APPLICATION_CREDENTIALS=${GOOGLE_APPLICATION_CREDENTIALS} \
  -e GCPF_TOKEN=${GCPF_TOKEN} \
  -e GCR_TOKEN=${GCR_TOKEN} \
  -e PYTHON_ENV=production \
  -v ${HOME}/.gcloud:/app/.gcloud \
  cfranklin11/tipresias_data_science:latest \
  npx sls deploy
