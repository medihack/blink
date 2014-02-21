#!/bin/bash

# workaround for SGE as it otherwise won't find blink_interface module
basedir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$basedir:$PYTHONPATH

$basedir/hcp_fconn_pipe.py -f subjects.txt
