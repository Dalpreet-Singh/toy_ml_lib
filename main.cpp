#include <iostream>
#include <memory>
#include <new>

class Matrix {
public:
  Matrix(int r, int c)
      : rows_(r), cols_(c)

  {
    data = new float[r * c];
    for (int i = 0; i < (r * c); i++) {
      data[i] = 0;
    }
  };

  ~Matrix() { delete[] data; };
  float &operator()(int r, int c) { return data[r * cols_ + c]; }
  int rows() { return rows_; }
  int cols() { return cols_; }
  void print() {
    for (int i = 0; i < (rows_ * cols_); i++) {
      std::cout << data[i] << ",";
    }
  }

private:
  int rows_;
  int cols_;
  float *data;
};
void fill(Matrix &a, float x) {
  for (int i = 0; i < a.rows(); i++) {
    for (int j = 0; j < a.cols(); j++) {
      a(i, j) = x;
    }
  }
}
Matrix &add(Matrix &out, Matrix &a, Matrix &b) {
  for (int i = 0; i < a.rows(); i++) {
    for (int j = 0; j < a.cols(); j++) {
      out(i, j) = a(i, j) + b(i, j);
    }
  }
  return out;
}

Matrix &matmul(Matrix &out, Matrix &a, Matrix &b) {

  for (int i = 0; i < a.rows(); i++) {
    for (int k = 0; k < a.cols(); k++) {
      for (int j = 0; j < b.cols(); j++) {
        out(i, j) += a(i, k) * b(k, j);
      }
    }
  }
  return out;
}

Matrix &sub(Matrix &out, Matrix &a, Matrix &b) {
  for (int i = 0; i < a.rows(); i++) {
    for (int j = 0; j < a.cols(); j++) {
      out(i, j) = a(i, j) - b(i, j);
    }
  }
  return out;
}
int main() {
  Matrix a(2, 3);
  Matrix b(3, 5);
  fill(a, 7);
  fill(b, 9);
  Matrix out(2, 5);
  matmul(out, a, b);
  out.print();

  return 0;
}
