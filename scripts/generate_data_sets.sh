#!/bin/bash

DATA_DIR="${PWD}/data"
mkdir -p "${DATA_DIR}/01_raw" "${DATA_DIR}/02_intermediate" "${DATA_DIR}/05_model_input"

DATA_IMPORT_DIR="${PWD}/src/augury/data_import"
python3 "${DATA_IMPORT_DIR}/betting_data.py"
python3 "${DATA_IMPORT_DIR}/match_data.py"
python3 "${DATA_IMPORT_DIR}/player_data.py"

kedro run --pipeline legacy -e development
kedro run --pipeline full -e development
