#!/bin/bash

if [ "$ENABLE_PROFILING" == "1" ]; then
    echo "Disabling line profiling..."
    unset ENABLE_PROFILING
else
    echo "Enabling line profiling..."
    export ENABLE_PROFILING="1"
fi

echo -e "\nCurrent value of ENABLE_PROFILING: $ENABLE_PROFILING"
