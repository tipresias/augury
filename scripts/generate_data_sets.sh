#!/bin/bash

mkdir -p "${PWD}/data/01_raw"

DATA_IMPORT_DIR="${PWD}/src/machine_learning/data_import"
python3 "${DATA_IMPORT_DIR}/betting_data.py"
python3 "${DATA_IMPORT_DIR}/match_data.py"
python3 "${DATA_IMPORT_DIR}/player_data.py"
