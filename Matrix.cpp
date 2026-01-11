

#include <iostream>
#include <memory>
#include <new>

#include "Matrix.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <source_location>
#include <string>
#include <vector>
[[noreturn]] void
t_error(std::string error_code,
        const std::source_location location = std::source_location::current()) {
  std::cerr << "Error: " << error_code << "\n"
            << "File: " << location.file_name() << "\n"
            << "Line: " << location.line() << "\n"
            << "Function: " << location.function_name() << std::endl;
  std::exit(EXIT_FAILURE);
}

Matrix::Matrix(int r, int c)
    : rows_(r), cols_(c), row_stride(c), col_stride(1), numel_(r * c)

{
  data = new float[numel_];
  for (int i = 0; i < (numel_); i++) {
    data[i] = 0;
  }
};

Matrix::Matrix(Matrix &&obj) {
  rows_ = obj.rows_;
  cols_ = obj.cols_;
  numel_ = obj.numel_;
  col_stride = obj.col_stride;
  row_stride = obj.row_stride;

  data = obj.data;
  obj.data = nullptr;
}
void Matrix::T() {
  std::swap(rows_, cols_);
  std::swap(row_stride, col_stride);
}

Matrix &Matrix::operator=(Matrix &&obj) {

  if (this == &obj)
    return *this;

  delete[] data;

  rows_ = obj.rows_;
  cols_ = obj.cols_;
  numel_ = obj.numel_;
  data = obj.data;

  col_stride = obj.col_stride;
  row_stride = obj.row_stride;
  obj.data = nullptr;
  return *this;
}

Matrix::~Matrix() { delete[] data; };
Matrix::Matrix(const Matrix &other)
    : rows_(other.rows_), cols_(other.cols_), numel_(other.numel_),
      row_stride(other.row_stride), col_stride(other.col_stride) {

  data = new float[numel_];
  for (int i = 0; i < numel_; i++) {
    data[i] = other.data[i];
  }
}

Matrix &Matrix::operator=(const Matrix &other) {
  if (this == &other)
    return *this;

  if (rows_ != other.rows_ || cols_ != other.cols_) {

    float *new_data = new float[other.numel_];
    delete[] data;
    data = new_data;
    rows_ = other.rows_;
    cols_ = other.cols_;
    numel_ = other.numel_;
  }
  col_stride = other.col_stride;
  row_stride = other.row_stride;

  for (int i = 0; i < numel_; i++) {
    data[i] = other.data[i];
  }
  return *this;
}

float &Matrix::operator()(int r, int c) {
  if (r < rows_ && c < cols_) {
    return data[r * row_stride + c * col_stride];
  }
  t_error("Matrix indexing is out of bounds!");
}
const float &Matrix::operator()(int r, int c) const {
  if (r < rows_ && c < cols_) {

    return data[r * row_stride + c * col_stride];
  }
  t_error("Matrix indexing is out of bounds!");
}

int Matrix::rows() const { return rows_; }
int Matrix::cols() const { return cols_; }
void Matrix::zero() {
  for (int i = 0; i < (numel_); i++) {
    data[i] = 0;
  }
}
void Matrix::print() const {
  for (int i = 0; i < (numel_); i++) {
    std::cout << data[i] << ",";
  }
}
int Matrix::numel() const { return numel_; }
void Matrix::shape() const { std::cout << "(" << rows_ << "," << cols_ << ")"; }

void fill(Matrix &a, float x) {
  for (int i = 0; i < a.rows(); i++) {
    for (int j = 0; j < a.cols(); j++) {
      a(i, j) = x;
    }
  }
}
Matrix &add(Matrix &out, const Matrix &a, const Matrix &b) {
  if (a.cols() == b.cols() && a.rows() == b.rows() && out.cols() == a.cols() &&
      out.rows() == a.rows()) {
    for (int i = 0; i < a.rows(); i++) {
      for (int j = 0; j < a.cols(); j++) {
        out(i, j) = a(i, j) + b(i, j);
      }
    }
    return out;
  } else if (a.cols() != b.cols() || a.rows() != b.rows()) {
    t_error("Matrix A and Matrix B do not have the same "
            "dimensions!");

  } else if (out.cols() != a.cols() || out.rows() != a.rows()) {
    t_error("Matrix Out does not have the same dimensions as Matrix A and B!");
  }
}
Matrix &sub(Matrix &out, const Matrix &a, const Matrix &b) {
  if (a.cols() == b.cols() && a.rows() == b.rows() && out.cols() == a.cols() &&
      out.rows() == a.rows()) {
    for (int i = 0; i < a.rows(); i++) {
      for (int j = 0; j < a.cols(); j++) {
        out(i, j) = a(i, j) - b(i, j);
      }
    }
    return out;
  } else if (a.cols() != b.cols() || a.rows() != b.rows()) {
    t_error("Matrix A and Matrix B do not have the same "
            "dimensions!");

  } else if (out.cols() != a.cols() || out.rows() != a.rows()) {
    t_error("Matrix Out does not have the same dimensions as Matrix A and B!");
  }
}

Matrix &matmul(Matrix &out, const Matrix &a, const Matrix &b) {
  if (a.cols() == b.rows() && out.rows() == a.rows() &&
      out.cols() == b.cols()) {
    out.zero();
    for (int i = 0; i < a.rows(); i++) {
      for (int k = 0; k < a.cols(); k++) {
        for (int j = 0; j < b.cols(); j++) {
          out(i, j) += a(i, k) * b(k, j);
        }
      }
    }
    return out;
  } else if (a.cols() != b.rows()) {

    t_error("Cols of Matrix A dont match up with rows of Matrix B");
  } else if (out.rows() != a.rows() || out.cols() != b.cols()) {
    t_error("Dimensions of out do not match up with rows of a and cols of b");
  }
}

void relu(Matrix &a) {
  for (int i = 0; i < a.rows(); i++) {
    for (int j = 0; j < a.cols(); j++) {
      a(i, j) = std::max(float(0), a(i, j));
    }
  }
}
