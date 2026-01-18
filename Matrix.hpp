
class alignas(16) Matrix {
public:
  Matrix(int r, int c);

  Matrix(Matrix &&obj);
  void T();
  Matrix T_C();

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
  Matrix scale(float x);
  void copy_raw_array_and_delete(float *data_c);

  Matrix sum_dim_0();

private:
  int rows_;
  int cols_;
  int numel_;
  int row_stride;
  int col_stride;
  float *data;
};

void fill(Matrix &a, float x);
Matrix add(const Matrix &a, const Matrix &b);
Matrix sub(const Matrix &a, const Matrix &b);
Matrix matmul(const Matrix &a, const Matrix &b);
Matrix broadcast_add(const Matrix &a, const Matrix &b);
Matrix &relu(Matrix &a);
Matrix d_relu(Matrix a);
Matrix hammard_product(const Matrix &a, const Matrix &b);
