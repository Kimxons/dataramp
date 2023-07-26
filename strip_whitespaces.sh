#!/bin/bash
find datahelp -type f -name "*.py" -exec sed -i 's/ *$//' {} \;