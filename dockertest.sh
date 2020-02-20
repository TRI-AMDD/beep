#!/usr/bin/env bash

# This script is run as the default procedure for testing
# in the docker container
echo "Version tag"
echo $BEEP_EP_VERSION_TAG

set -e

# Set TQDM to be off in tests
export TQDM_OFF=1

# Run nosetests
nosetests --with-xunit --all-modules --traverse-namespace \
    --with-coverage --cover-package=beep_ep --cover-inclusive

# Generate coverage
python -m coverage xml --include=beep_ep*

# Do linting
pylint -f parseable -d I0011,R0801 beep_ep | tee pylint.out

