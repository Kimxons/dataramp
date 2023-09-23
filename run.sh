#!/bin/bash

LOG_FILE="dh_logs/deployment.log"
CUSTOM_LOGGER_MODULE="custom_logger.py"
PYTHON_BIN="python3"

log() {
  echo "$(date "+%Y-%m-%d %H:%M:%S"): $1" >> "$LOG_FILE"
}

initialize_custom_logger() {
  if [ -f "$CUSTOM_LOGGER_MODULE" ]; then
    log "Initializing custom logger..."
    $PYTHON_BIN -c "from custom_logger import Logger; logger = Logger(logger_name='deployment', filename='$LOG_FILE')"
  else
    log "Custom logger module '$CUSTOM_LOGGER_MODULE' not found. Skipping custom logging."
  fi
}

close_custom_logger() {
  if [ -f "$CUSTOM_LOGGER_MODULE" ]; then
    log "Closing custom logger..."
    $PYTHON_BIN -c "from custom_logger import Logger; logger = Logger(logger_name='deployment', filename='$LOG_FILE'); logger.close()"
  fi
}

cleanup() {
  close_custom_logger
  exit
}

trap cleanup EXIT

log "Deployment started."

log "Checking required packages..."
$PYTHON_BIN check_requirements.py requirements_dev.txt

if [[ $? -eq 1 ]]; then
    log "Installing missing packages..."
    $PYTHON_BIN -m pip install -r requirements_dev.txt
fi

initialize_custom_logger

log "Deployment completed successfully."
