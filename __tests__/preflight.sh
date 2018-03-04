#!/bin/bash -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set up modified times for the test files to match the expect cases.
echo -n "Setting modified times for test files..."
touch -t 201801152235.10 "${DIR}/cases/test1/file.1"
touch -t 201801152235.18 "${DIR}/cases/test1/directory1/file.2"
echo "done"
