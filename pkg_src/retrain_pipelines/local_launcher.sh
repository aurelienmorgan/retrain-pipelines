#!/bin/bash
export no_proxy=localhost,0.0.0.0
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

export PARENT_DIR=$(pwd)

export METAFLOW_SERVICE_URL=${METAFLOW_SERVICE_URL:-http://localhost:8080/}
export METAFLOW_DEFAULT_METADATA=service
export METAFLOW_DEFAULT_DATASTORE=local
export METAFLOW_DATASTORE_SYSROOT_LOCAL=\
${METAFLOW_DATASTORE_SYSROOT_LOCAL:-${HOME}/local_datastore/}

#---------------------------------------------------------------

# Create local_datastore dir if first launch (with owner : USER)
(
  umask 0022
  mkdir -p "$METAFLOW_DATASTORE_SYSROOT_LOCAL"
)

#---------------------------------------------------------------

# echo $(which python)

# Check if retrain_pipelines Python package is installed
# otherwise add "pkg_src" to PYTHONPATH
# if exists and not there already.
if ! python3 -c "import retrain_pipelines" &> /dev/null; then
    # "'retrain_pipelines' package is not installed."
    PARENT_PARENT_DIR="$(dirname "$(dirname "${BASH_SOURCE[0]}")")"
    if [[ "$(basename "$PARENT_PARENT_DIR")" == "pkg_src" ]]; then
        # "'pkg_src' directory exists."
        if [[ ":$PYTHONPATH:" != *":$PARENT_PARENT_DIR:"* ]]; then
            # ./pkg_src is not already in PYTHONPATH
            export PYTHONPATH="$PARENT_PARENT_DIR:$PYTHONPATH"
            #echo $PYTHONPATH
        fi
    else
        # 'pkg_src' directory does not exist."
        RED='\033[0;31m' # Define red color
        NC='\033[0m'     # No color (reset)
        echo -e "${RED}Couldn't find a 'retrain_pipelines' installation.${NC}"
        exit 1
    fi
else
    : # 'retrain_pipelines' package is already installed."
fi

#---------------------------------------------------------------


usage() {
  echo "Usage: $0 <file_path> <run|resume> [options]"
  echo ""
  echo "Arguments:"
  echo "  file_path   Path to the Python file to execute"
  echo "  run         Execute the script from start."
  echo "  resume      Resume the execution from step to be named."
  echo "Options:"
  echo "  --help      for details"
  exit 1
}

if [[ $# -lt 2 ]]; then
  echo "Error: Not enough arguments."
  usage
  exit 1
fi

# The first argument is the file path
FILE_PATH=$1
# Check if FILE_PATH is a relative path
# and prepend $PARENT_DIR if necessary
if [[ "$FILE_PATH" != /* ]]; then
  FILE_PATH="'${PARENT_DIR}/${FILE_PATH}'"
else
  FILE_PATH="'$1'"
fi

# The second argument is the metaflow execution mode (run vs. resume)
EXECUTION_MODE=$2

if [[ "$EXECUTION_MODE" != "run" &&
      "$EXECUTION_MODE" != "resume" ]]; then
  echo "Error: Invalid execution mode '$EXECUTION_MODE'. " \
       "Must be 'run', or 'resume'."
  usage
  exit 1
fi

# Shift the positional parameters to remove the first two arguments
# (FILE_PATH and EXECUTION_MODE)
shift 2

# Initialize variables
DATA_FILE=""
PREPROCESS_ARTIFACTS_PATH=""
PIPELINE_CARD_ARTIFACTS_PATH=""
OPTIONAL_ARGS=()

# Loop through remaining arguments to handle special cases
while [[ $# -gt 0 ]]; do
  case $1 in
    --data_file)
      if [[ -n "$2" ]]; then
        DATA_FILE=$2
        shift # Skip the argument value
        # Check if DATA_FILE is a relative path
        # and prepend $PARENT_DIR if necessary
        if [[ "$DATA_FILE" != /* ]]; then
          DATA_FILE="'${PARENT_DIR}/${DATA_FILE}'"
        else
          DATA_FILE="'${DATA_FILE}'"
        fi
      else
        echo "Error: --data_file requires a value"
        exit 1
      fi
      ;;
    --preprocess_artifacts_path)
      if [[ -n "$2" ]]; then
        PREPROCESS_ARTIFACTS_PATH=$2
        shift # Skip the argument value
        # Check if PREPROCESS_ARTIFACTS_PATH is a relative path
        # and prepend $PARENT_DIR if necessary
        if [[ "$PREPROCESS_ARTIFACTS_PATH" != /* ]]; then
          PREPROCESS_ARTIFACTS_PATH="'${PARENT_DIR}/${PREPROCESS_ARTIFACTS_PATH}'"
        fi
      else
        echo "Error: --preprocess_artifacts_path requires a value"
        exit 1
      fi
      ;;
    --pipeline_card_artifacts_path)
      if [[ -n "$2" ]]; then
        PIPELINE_CARD_ARTIFACTS_PATH=$2
        shift # Skip the argument value
        # Check if PIPELINE_CARD_ARTIFACTS_PATH is a relative path
        # and prepend $PARENT_DIR if necessary
        if [[ "$PIPELINE_CARD_ARTIFACTS_PATH" != /* ]]; then
          PIPELINE_CARD_ARTIFACTS_PATH="'${PARENT_DIR}/${PIPELINE_CARD_ARTIFACTS_PATH}'"
        fi
      else
        echo "Error: --pipeline_card_artifacts_path requires a value"
        exit 1
      fi
      ;;
    --*)
      # Handle optional arguments by enclosing their values in single quotes
      OPTIONAL_ARGS+=("$1 '$(echo $2)'")
      shift # Skip the argument value
      ;;
    *)
      # Pass optional arguments that don't start with "--" as they are
      # (unnamed, just value)
      OPTIONAL_ARGS+=("'$1'")
      ;;
  esac
  shift # Move to the next argument
done

# Change to the datastore directory
cd $METAFLOW_DATASTORE_SYSROOT_LOCAL

# Construct the command
COMMAND="python ${FILE_PATH} ${EXECUTION_MODE}"
if [[ -n "$DATA_FILE" ]]; then
  COMMAND="${COMMAND} --data_file ${DATA_FILE}"
fi
if [[ -n "$PREPROCESS_ARTIFACTS_PATH" ]]; then
  COMMAND="${COMMAND} --preprocess_artifacts_path ${PREPROCESS_ARTIFACTS_PATH}"
fi
if [[ -n "$PIPELINE_CARD_ARTIFACTS_PATH" ]]; then
  COMMAND="${COMMAND} --pipeline_card_artifacts_path ${PIPELINE_CARD_ARTIFACTS_PATH}"
fi
COMMAND="${COMMAND} ${OPTIONAL_ARGS[@]}"

# Execute the Python script with the constructed command
#echo $COMMAND
eval $COMMAND

# sed -i 's/\r//' local_launcher.sh
# chmod +x local_launcher.sh
