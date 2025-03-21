#!/bin/bash

DIR="$( cd "$( dirname "$(readlink -f "${BASH_SOURCE[0]}")" )" && pwd )"

RED='\033[0;31m'
RESET='\033[0m'


if [[ ! $# -eq 2 ]]; then
echo -e "${RED}[x] error: expecting $0 <CONDA_ENV_NAME> <CONDA_ENV_DIR>!${RESET}"
exit -1
fi

export CONDA_NAME=$1
export CONDA_PATH=$2
export GXX_COMPILER_VERSION="12.2.0"
ARTISAN_DIR=$(cd / && $CONDA_PATH/bin/python -c "try:
    import os.path as osp, artisan
    print(osp.dirname(artisan.__file__))
except ImportError:
    print('')")

# Check if the Python command failed
if [ $? -ne 0 ] || [ -z "$ARTISAN_DIR" ]; then
    echo -e "${RED}[x] error: 'artisan' module not installed in this conda environment!${RESET}"
    exit -1
fi

export ARTISAN_DIR

mkdir -p $CONDA_PATH/lib/nvvm/libdevice
cp $CONDA_PATH/lib/libdevice.10.bc $CONDA_PATH/lib/nvvm/libdevice/

envsubst '$CONDA_NAME $CONDA_PATH $GXX_COMPILER_VERSION $ARTISAN_DIR' < $DIR/activate.template > $CONDA_PATH/etc/conda/activate.d/${CONDA_NAME}_activate.sh
envsubst '$CONDA_NAME $CONDA_PATH $GXX_COMPILER_VERSION $ARTISAN_DIR' < $DIR/deactivate.template > $CONDA_PATH/etc/conda/deactivate.d/${CONDA_NAME}_deactivate.sh
