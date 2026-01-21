
#include "Matrix.hpp"
#include "utility.hpp"

class Linear {
public:
  Linear(int in_features, int out_features, int od);

  Matrix forward(const Matrix &input);
  void init();
  Matrix backward(const Matrix &grad_output);
  int get_order();
  const Matrix &get_outputs() const;
  Matrix &get_weights();
  Matrix get_grad_weights() const;
  Matrix get_grad_biases() const;
  Matrix &get_biases();

private:
  Matrix weights;
  Matrix bias;
  Matrix inputs;
  Matrix outputs;
  int order;
  Matrix grad_weights;
  Matrix grad_biases;
};

class model {
public:
  model(std::initializer_list<Linear *> list);
  ~model();
  Matrix forward(const Matrix &input);
  void init();
  void backward(const Matrix &true_prob, const Matrix &pred_prob);

  void step(float lr);

private:
  std::vector<Linear *> layers;
};
void training(model &a, std::string train_file, std::string train_label_file,
              int batch_size, int rows, int cols, int epochs);
void eval(model &a, std::string test_file, std::string test_label_file,
          int batch_size, int rows, int cols);
