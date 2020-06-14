#!/bin/bash

set -euo pipefail

PORT=8008

# We need travis_wait for Travis CI builds, because installing R packages takes forever.
# On a local machine, we can just build & deploy as normal.
if [ -n "$(LC_ALL=C type -t travis_wait)" ] && [ "$(LC_ALL=C type -t travis_wait)" = function ]
then
  travis_wait 30 gcloud builds submit --config cloudbuild.yaml
else
  gcloud builds submit --config cloudbuild.yaml
fi

GOOGLE_ENV_VARS="
AFL_DATA_SERVICE_TOKEN=${AFL_DATA_SERVICE_TOKEN},\
PYTHON_ENV=production,\
DATA_SCIENCE_SERVICE_TOKEN=${DATA_SCIENCE_SERVICE_TOKEN},\
AFL_DATA_SERVICE=${AFL_DATA_SERVICE},\
TIPRESIAS_APP_TOKEN=${TIPRESIAS_APP_TOKEN}
"

gcloud beta run deploy augury \
  --image gcr.io/${PROJECT_ID}/augury \
  --memory 2048Mi \
  --region us-central1 \
  --platform managed \
  --update-env-vars ${GOOGLE_ENV_VARS}

if [ $? != 0 ]
then
  exit $?
fi


APP_DIR=/var/www/${PROJECT_NAME}
DOCKER_IMAGE=cfranklin11/tipresias_data_science:latest

sudo chmod 600 ~/.ssh/deploy_rsa
sudo chmod 755 ~/.ssh

docker pull ${DOCKER_IMAGE}
docker build --cache-from ${DOCKER_IMAGE} -t ${DOCKER_IMAGE} .
docker push ${DOCKER_IMAGE}

RUN_APP="
  cd ${APP_DIR} \
    && docker pull ${DOCKER_IMAGE} \
    && docker stop ${PROJECT_NAME}_app \
    && docker container rm ${PROJECT_NAME}_app \
    && docker run \
      -d \
      -v ${APP_DIR}/.gcloud:/app/.gcloud \
      --env-file .env \
      -p ${PORT}:${PORT} \
      -e PYTHON_ENV=production \
      -e GOOGLE_APPLICATION_CREDENTIALS=.gcloud/keyfile.json \
      -e PYTHONPATH=./src \
      --name ${PROJECT_NAME}_app \
      ${DOCKER_IMAGE}
"

# We use 'ssh' instead of 'doctl compute ssh' to be able to bypass key checking.
ssh -i ~/.ssh/deploy_rsa -oStrictHostKeyChecking=no \
  ${DIGITAL_OCEAN_USER}@${PRODUCTION_HOST} \
  ${RUN_APP}

if [ $? != 0 ]
then
  exit $?
fi

./scripts/wait-for-it.sh ${PRODUCTION_HOST}:${PORT} \
  -t 60 \
  -- ./scripts/post_deploy.sh

exit $?
