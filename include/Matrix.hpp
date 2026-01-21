#pragma once
class Matrix {
public:
  Matrix(int r, int c);

  Matrix(Matrix &&obj);
  void T();
  Matrix T_C() const;

  Matrix &operator=(Matrix &&obj);
  ~Matrix();
  Matrix(const Matrix &other);
  Matrix &operator=(const Matrix &other);
  float &operator()(int r, int c);
  const float &operator()(int r, int c) const;
  int rows() const;
  int cols() const;
  void zero();
  void print() const;
  int numel() const;
  void shape() const;
  Matrix scale_copy(const float &x) const;
  Matrix &scale_inplace(const float &x);

  void copy_raw_array_and_delete(float *data_c);
  Matrix slice_rows(int row_start, int num_rows);
  Matrix sum_dim_0_copy() const;
  void row_copy(const Matrix &vec, int row_start);

private:
  int rows_;
  int cols_;
  int numel_;
  int row_stride;
  int col_stride;
  float *data = nullptr;
};

void fill(Matrix &a, float x);
Matrix add(const Matrix &a, const Matrix &b);
Matrix sub(const Matrix &a, const Matrix &b);
Matrix matmul(const Matrix &a, const Matrix &b);
Matrix broadcast_add(const Matrix &a, const Matrix &b);
Matrix &relu_inplace(Matrix &a);
Matrix relu_copy(const Matrix &a);
Matrix d_relu_copy(const Matrix &a);
Matrix &d_relu_inplace(Matrix &a);
Matrix hammard_product(const Matrix &a, const Matrix &b);
Matrix &softmax(Matrix &a);
