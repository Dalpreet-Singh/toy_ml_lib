

#include <iostream>

#include "Matrix.hpp"
#include "utility.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <random>
#include <source_location>
#include <string>
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> dist(0.044f, 0.107f);
float rand_num() { return dist(gen); }

Matrix::Matrix(int r, int c)
    : rows_(r), cols_(c), row_stride(c), col_stride(1), numel_(r * c)

{
  data = std::vector<float>(numel_);
};

Matrix::Matrix(Matrix &&obj) {
  rows_ = obj.rows_;
  cols_ = obj.cols_;
  numel_ = obj.numel_;
  col_stride = obj.col_stride;
  row_stride = obj.row_stride;

  data = std::move(obj.data);
  obj.rows_ = 0;
  obj.cols_ = 0;
  obj.numel_ = 0;
  obj.col_stride = 0;
  obj.row_stride = 0;
}
void Matrix::T() {

  if (data.empty()) {
    t_error("Transpose called on invalid matrix");
  }
  std::swap(rows_, cols_);
  std::swap(row_stride, col_stride);
}
Matrix Matrix::T_C() const {

  if (data.empty()) {
    t_error("Transpose called on invalid matrix");
  }
  Matrix copy = *this;
  std::swap(copy.rows_, copy.cols_);
  std::swap(copy.row_stride, copy.col_stride);
  return copy;
}
void Matrix::row_copy(const Matrix &vec, int row_start) {
  if (vec.cols_ == cols_ && vec.rows() + row_start < rows_ && row_start >= 0)
    for (int i = row_start; i < row_start + vec.rows(); i++) {
      for (int j = 0; j < cols_; j++) {
        (*this)(i, j) = vec(i - row_start, j);
      }
    }
  else {
    t_error("Invalid row_start or invalid copy matrix dimensions");
  }
}

Matrix &Matrix::operator=(Matrix &&obj) {

  if (this == &obj)
    return *this;

  rows_ = obj.rows_;
  cols_ = obj.cols_;
  numel_ = obj.numel_;
  data = std::move(obj.data);

  col_stride = obj.col_stride;
  row_stride = obj.row_stride;
  obj.cols_ = 0;
  obj.numel_ = 0;
  obj.col_stride = 0;
  obj.row_stride = 0;

  return *this;
}

Matrix::~Matrix() = default;
Matrix::Matrix(const Matrix &other)
    : rows_(other.rows_), cols_(other.cols_), numel_(other.numel_),
      row_stride(other.row_stride), col_stride(other.col_stride) {

  data = std::vector<float>(numel_);
  for (int i = 0; i < numel_; i++) {
    data[i] = other.data[i];
  }
}
void Matrix::copy_raw_array_and_delete(float *data_c) {
  for (int i = 0; i < numel_; i++) {
    data[i] = data_c[i];
  }
  delete[] data_c;
}
Matrix Matrix::slice_rows(int row_start, int num_rows) {
  if (row_start >= 0 && row_start + num_rows <= rows_) {
    Matrix out(num_rows, cols_);
    for (int i = row_start; i < row_start + num_rows; i++) {
      for (int j = 0; j < cols_; j++) {
        out(i - row_start, j) = (*this)(i, j);
      }
    }
    return out;
  } else {
    t_error("Row_start and num_rows are invalid for row slicing");
  }
}

Matrix &Matrix::operator=(const Matrix &other) {
  if (this == &other)
    return *this;

  if (rows_ != other.rows_ || cols_ != other.cols_) {

    rows_ = other.rows_;
    cols_ = other.cols_;
    numel_ = other.numel_;
  }
  data = other.data;
  col_stride = other.col_stride;
  row_stride = other.row_stride;

  return *this;
}

float &Matrix::operator()(int r, int c) {
  if (r < rows_ && r > -1 && c > -1 && c < cols_) {
    return data[r * row_stride + c * col_stride];
  }
  t_error("Matrix indexing is out of bounds!");
}
const float &Matrix::operator()(int r, int c) const {
  if (r < rows_ && r > -1 && c > -1 && c < cols_) {

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
Matrix Matrix::sum_dim_0_copy() const {
  if (rows_ > 0 && cols_ > 0) {
    Matrix out(1, this->cols());
    for (int i = 0; i < this->cols(); i++) {
      float sum = 0;
      for (int j = 0; j < this->rows(); j++) {
        sum += (*this)(j, i);
      }
      out(0, i) = sum;
    }
    return out;
  } else {
    t_error("Doing a summing op on a matrix with 0 numel is undefined");
  }
}
Matrix Matrix::scale_copy(const float &x) const {
  Matrix copy = (*this);
  for (int i = 0; i < copy.rows_; i++) {
    for (int j = 0; j < copy.cols_; j++) {
      copy(i, j) *= x;
    }
  }
  return copy;
}
Matrix &Matrix::scale_inplace(const float &x) {

  for (int i = 0; i < this->rows_; i++) {
    for (int j = 0; j < this->cols_; j++) {
      (*this)(i, j) *= x;
    }
  }
  return *this;
}

void Matrix::print() const {
  std::cout << "(" << std::endl;
  for (int i = 0; i < (rows_); i++) {
    for (int j = 0; j < cols_; j++) {
      std::cout << (*this)(i, j) << ",";
    }
    std::cout << std::endl;
  }
  std::cout << ")" << std::endl;
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
Matrix add(const Matrix &a, const Matrix &b) {
  if (a.cols() == b.cols() && a.rows() == b.rows()) {
    Matrix out(a.rows(), a.cols());
    for (int i = 0; i < a.rows(); i++) {
      for (int j = 0; j < a.cols(); j++) {
        out(i, j) = a(i, j) + b(i, j);
      }
    }
    return out;
  } else {
    t_error("Matrix A and Matrix B do not have the same "
            "dimensions!");
    return Matrix(0, 0);
  }
}

Matrix sub(const Matrix &a, const Matrix &b) {
  if (a.cols() == b.cols() && a.rows() == b.rows()) {
    Matrix out(a.rows(), a.cols());
    for (int i = 0; i < a.rows(); i++) {
      for (int j = 0; j < a.cols(); j++) {
        out(i, j) = a(i, j) - b(i, j);
      }
    }
    return out;
  } else {
    t_error("Matrix A and Matrix B do not have the same "
            "dimensions!");
    return Matrix(0, 0);
  }
}
Matrix matmul(const Matrix &a, const Matrix &b) {
  if (a.cols() == b.rows()) {
    Matrix out(a.rows(), b.cols());
    out.zero();

    for (int i = 0; i < a.rows(); i++) {
      for (int k = 0; k < a.cols(); k++) {
        for (int j = 0; j < b.cols(); j++) {
          out(i, j) += a(i, k) * b(k, j);
        }
      }
    }
    return out;
  } else {

    t_error("Cols of Matrix A dont match up with rows of Matrix B");
    return Matrix(0, 0);
  }
}
Matrix broadcast_add(const Matrix &a, const Matrix &b) {

  if (a.cols() == b.cols() && b.rows() == 1) {
    Matrix out(a.rows(), a.cols());
    for (int i = 0; i < a.rows(); i++) {
      for (int j = 0; j < b.cols(); j++) {
        out(i, j) = a(i, j) + b(0, j);
      }
    }
    return out;
  } else {
    t_error("dimensions are not valid for a broadcasted add");
    return Matrix(0, 0);
  }
}

Matrix &relu_inplace(Matrix &a) {
  for (int i = 0; i < a.rows(); i++) {
    for (int j = 0; j < a.cols(); j++) {
      a(i, j) = std::max(float(0), a(i, j));
    }
  }
  return a;
}
Matrix relu_copy(const Matrix &a) {
  Matrix out = a;
  for (int i = 0; i < a.rows(); i++) {
    for (int j = 0; j < a.cols(); j++) {
      out(i, j) = std::max(float(0), out(i, j));
    }
  }
  return out;
}
Matrix &d_relu_inplace(Matrix &a) {
  for (int i = 0; i < a.rows(); i++) {
    for (int j = 0; j < a.cols(); j++) {
      a(i, j) = (a(i, j) <= 0) ? 0 : 1;
    }
  }
  return a;
}
Matrix d_relu_copy(const Matrix &a) {
  Matrix out = a;
  for (int i = 0; i < a.rows(); i++) {
    for (int j = 0; j < a.cols(); j++) {
      out(i, j) = (out(i, j) <= 0) ? 0 : 1;
    }
  }
  return out;
}
Matrix hammard_product(const Matrix &a, const Matrix &b) {
  if (a.cols() == b.cols() && a.rows() == b.rows()) {
    Matrix out(a.rows(), a.cols());
    for (int i = 0; i < a.rows(); i++) {
      for (int j = 0; j < a.cols(); j++) {
        out(i, j) = a(i, j) * b(i, j);
      }
    }
    return out;
  } else {
    t_error("Matrix A and Matrix B do not have the same "
            "dimensions!");
    return Matrix(0, 0);
  }
}
Matrix &softmax(Matrix &a) {
  if (a.cols() <= 0) {
    t_error("You must have atleast 1 col for softmax");
  }
  for (int i = 0; i < a.rows(); i++) {
    float max_val = a(i, 0);
    for (int j = 0; j < a.cols(); j++) {
      max_val = std::max(a(i, j), max_val);
    }
    float sum = 0;
    for (int j = 0; j < a.cols(); j++) {
      a(i, j) = exp(a(i, j) - max_val);
      sum += a(i, j);
    }
    for (int j = 0; j < a.cols(); j++) {
      a(i, j) /= sum;
    }
  }
  return a;
}
