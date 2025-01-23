#!/bin/bash

find dataramp -type f -name "*.py" -exec sed -i 's/ *$//' {} \;
