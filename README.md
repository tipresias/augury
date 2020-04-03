# Augury

[![Build Status](https://travis-ci.com/tipresias/augury.svg?branch=master)](https://travis-ci.com/tipresias/augury)
[![Maintainability](https://api.codeclimate.com/v1/badges/e3f72dba19bb5f121622/maintainability)](https://codeclimate.com/github/tipresias/augury/maintainability)
[![Test Coverage](https://api.codeclimate.com/v1/badges/e3f72dba19bb5f121622/test_coverage)](https://codeclimate.com/github/tipresias/augury/test_coverage)

Jupyter Notebooks and machine-learning service for the Tipresias app

## Running things

### Setup

- To manage environemnt variables:
    - Install [`direnv`](https://direnv.net/)
    - Add `eval "$(direnv hook bash)"` to the bottom of `~/.bashrc`
    - Run `direnv allow .` inside the project directory
- To build and run the app: `docker-compose up --build`

### Run the app

- `docker-compose up`
- Navigate to `localhost:8008`.

### Run Jupyter notebook in Docker

- If it's not already running, run Jupyter with `docker-compose up notebook`.
- The terminal will display something like the following message:

```
notebook_1  | [I 03:01:38.909 NotebookApp] The Jupyter Notebook is running at:
notebook_1  | [I 03:01:38.909 NotebookApp] http://(ea7b71b85805 or 127.0.0.1):8888/?token=dhf7674ururrtuuf8968lhhjdrfjghicty57t69t85e6dhfj
notebook_1  | [I 03:01:38.909 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
```

- Copy the URL given and paste it into your browser.
- Alternatively, copy just the token from the terminal, navigate your browser to `http://localhost:8888`, and paste the token into the given field.

### Testing

#### Run Python tests

- `docker-compose run --rm data_science kedro test --no-cov`
  - Note: Remove `--no-cov` to generate a test-coverage report in the terminal.
- Linting: `docker-compose run --rm <python service> pylint --disable=R <python modules you want to lint>`
  - Note: `-d=R` disables refactoring checks for quicker, less-opinionated linting. Remove that option if you want to include those checks.

### Deploy

- `augury` is deployed to Google Cloud via Travis CI. See `scripts/deploy.sh` for specific commands.

## Troubleshooting

- When working with some of the larger data sets (e.g. player stats comprise over 600,000 rows), your process might mysteriously die without completing. This is likely due to Docker running out of memory, because the default 2GB isn't enough. At least 4GB is the recommended limit, but you'll want more if you plan on having multiple processes running or multiple Jupyter notebooks open at the same time.
