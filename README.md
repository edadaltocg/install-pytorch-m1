# Install PyTorch natively on Mac M1

## Install homebrew on mac m1

Skip this part if you already have installed ```homebrew``` on your machine.

    xcode-select --install
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

## Basic requirements

### Install CMake

    brew install cmake

### Install C++ compiler

    brew install gcc

### Install libbffi

    brew install libffi

## Install latest python

Skip this part if have it already installed.

    brew install python
    pip3 install --upgrade pip setuptools wheel
    brew install numpy

Create a virtual environment.

    virtualenv venv --system-site-packages

Activate the local environment.

    source env/bin/activate

Check that you have the correct python ready

    which python && python -V

## Install PyTorch natively

### Build pytorch from source

    git clone http://github.com/pytorch/pytorch
    cd pytorch
    git submodule sync
    git submodule update --init --recursive
    pip install -r requirements.txt
    python setup.py develop

And voila! Run the scripts ```benchmark.py``` and ```profiler.py``` to check the performance of your system.

    python benchmark.py
    python profiler.py

Open ```trace.json``` at ```chrome://tracing/```
