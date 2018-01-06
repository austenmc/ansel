#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo { \"remote\": `cat "$DIR/remote.sample"`, \"local\": `cat "$DIR/local.sample"` }
