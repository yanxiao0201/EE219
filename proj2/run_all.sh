#!/bin/bash

if [[ $# -lt 1 ]]; then
	echo "Usage: ./run_all.sh ls_pattern"
	exit 0
fi

if [[ $1 =~ .*\.\.\/.* || $1 =~ .*\/\.\..* ]]; then
	echo "Please stay within current directory..."
	exit 0
fi

for file in `ls $1`; do
	echo "========Running $file...========"
	python $file
	if [[ $? -ne 0 ]]; then
		echo "Aborting execution for the rest of the scripts in the list..."
		break
	fi
done
