# Toy Machine Learning Library

A small toy machine learning library written in C++, built primarily as a learning project. The goal is to understand how core machine learning components work under the hood with minimal abstraction.

## Features

* Support for up to 2D matrices
* Standard matrix operations (e.g. matrix multiplication, element-wise operations)
* A Linear layer and Model abstraction for training and inference
* Softmax, cross-entropy loss, and SGD optimizer
* Utility functions for loading datasets stored as `float32` binary files
* Example dataset loader for CIFAR-10

## Project Goals

* Learn how neural networks are implemented from scratch
* Avoid external machine learning frameworks
* Keep the code simple, explicit, and easy to reason about

## Getting Started


### Clone the Repository

```bash
git clone https://github.com/Dalpreet-Singh/toy_ml_lib.git
cd toy_ml_lib
```

### Prepare Your Dataset

You can either:

* Convert your dataset into a `float32` binary format
* Use `scripts/data.py` to load and preprocess your data

An example CIFAR-10 loader is already provided.

### Create Your Model

* Define an MLP by initializing a `Model` class with instances of `Linear` classes(a standard CIFAR-10 model is already provided in main.cpp)
* The activation function is currently hardcoded to ReLU

### Train the Model

After loading your data, call:

```cpp
training_loop(...)
```

from the `main()` function in your source file.

## Building

This project uses CMake.

### Configure the Executable

Open `CMakeLists.txt` and change:

```cmake
add_executable(train main.cpp)
```

to:

```cmake
add_executable(train YOUR_FILE.cpp)
```

⚠️ **Warning:** Ensure `YOUR_FILE.cpp` is not located inside any subdirectories.

### Build the Project

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Run the Executable

```bash
./build/train
```

## Notes

* This library is not optimized
* Numerical stability is limited
* Intended strictly for learning and experimentation
* Expect rough edges 🙂
