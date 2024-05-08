#!/bin/bash

LIB_NAME="PFRTool"

if pip list | grep -F $LIB_NAME; then
    echo "$LIB_NAME is already installed. Upgrading..."
    pip install . --upgrade
else
    echo "$LIB_NAME is not installed. Installing..."
    pip install .
fi
