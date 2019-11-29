#!/bin/bash

set -euo pipefail

# We use loadfile to group tests by file to avoid IO errors
docker-compose run --rm data_science kedro test -n auto --dist=loadfile $*
