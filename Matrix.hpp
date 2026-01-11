
class alignas(16) Matrix {
public:
  Matrix(int r, int c);

  Matrix(Matrix &&obj);
  void T();

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

private:
  int rows_;
  int cols_;
  int numel_;
  int row_stride;
  int col_stride;
  float *data;
};

void fill(Matrix &a, float x);
Matrix &add(Matrix &out, const Matrix &a, const Matrix &b);
Matrix &sub(Matrix &out, const Matrix &a, const Matrix &b);
Matrix &matmul(Matrix &out, const Matrix &a, const Matrix &b);
void relu(Matrix &a);
