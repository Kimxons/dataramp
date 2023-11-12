#!/bin/bash
set -e

# Manual steps to build and upload a new package version:
# conda build . --python <python version>
# conda build prints output path of the .tar.bz2 file
# conda convert --platform all <path of tar.bz2 file>  -o $HOME/Anaconda3/conda-bld/
# anaconda login --username <username> --password <password>
# anaconda upload <path of tar.bz2 file> (have to do that for each tar file)
# anaconda logout

PYTHON_VERSIONS=(3.7 3.8 3.9 3.10)
PLATFORMS=(osx-64 linux-32 linux-64 win-32 win-64)

anaconda_credentials=($(cat "anaconda_credentials.txt"))
ANACONDA_USERNAME="${anaconda_credentials[0]}"
ANACONDA_ORGANIZATION="${anaconda_credentials[1]}"
ANACONDA_PASSWORD="${anaconda_credentials[2]}"

for py_version in "${PYTHON_VERSIONS[@]}"; do
    conda build . --python "$py_version"
done

bit=$(uname -m)
if [[ "$bit" == *"x86_64"* ]]; then
    bit="64"
else
    bit="32"
fi

case "$OSTYPE" in
    "linux"*) host_platform="linux-$bit" ;;
    "darwin"*) host_platform="osx-$bit" ;;
    "cygwin"*) host_platform="win-$bit" ;;
    "msys"*) host_platform="win-$bit" ;;
    "win"*) host_platform="win-$bit" ;;
    *)
        echo "ERROR: Unknown platform!"
        exit 1
        ;;
esac

cd ~
if [ -d ./Anaconda3 ]; then
    conda_build_path="Anaconda3/conda-bld"
elif [ -d ./miniconda3 ]; then
    conda_build_path="miniconda3/conda-bld"
else
    echo "ERROR: Could not determine directory of conda-build!"
    exit 1
fi

for file in $(find "$HOME/$conda_build_path/$host_platform" -name *.tar.bz2); do
    for platform in "${PLATFORMS[@]}"; do
        conda convert --platform "$platform" "$file" -o "$HOME/Anaconda3/conda-bld/"
    done
done

anaconda login --username "$ANACONDA_USERNAME" --password "$ANACONDA_PASSWORD"

for file in $(find "$HOME/$conda_build_path/" -name *.tar.bz2); do
    echo "$file"
    anaconda upload "$file"
done

anaconda logout

echo "Execution completed!"
