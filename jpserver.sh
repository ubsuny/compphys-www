#!/bin/bash
if [ -z $1 ]; then
	JPORT=8990
else
	JPORT="${1}"
fi
jupyter-lab  --port $JPORT --no-browser
