#!/bin/bash

set -euo pipefail

docker-compose run --rm data_science sls deploy
