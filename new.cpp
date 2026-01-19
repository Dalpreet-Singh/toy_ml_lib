

#include "Arena.hpp"
#include "Matrix.hpp"
#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <vector>
class Linear {
public:
  Linear(int in_features, int out_features, int od)
      : weights(out_features, in_features), inputs(Matrix(0, 0)),
        bias(1, out_features), order(od), grad_weights(0, 0), grad_biases(0, 0),
        outputs(Matrix(0, 0)) {}

  Matrix forward(const Matrix &input) {
    inputs = input;

    // input shape (b,i)
    // weight shape (o,i)
    // bias shape (o)

    Matrix output = broadcast_add((matmul(input, weights.T_C())), bias);
    if (order != -1) {
      relu_inplace(output);
    }
    outputs = output;

    return outputs;
  }
  Matrix backward(const Matrix &grad_output) {

    // inputs(b,i)
    grad_biases = grad_output.sum_dim_0_copy();
    grad_weights = matmul(grad_output.T_C(), inputs);
    return matmul(grad_output, weights);
  }
  int get_order() { return order; }
  const Matrix &get_outputs() const { return outputs; }
  Matrix &get_weights() { return weights; }
  Matrix get_grad_weights() const { return grad_weights; }
  Matrix get_grad_biases() const { return grad_biases; }
  Matrix &get_biases() { return bias; }

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
  model(std::initializer_list<Linear *> list) {
    for (Linear *layer : list) {
      layers.push_back(layer);
    }
  }
  ~model() {
    for (Linear *layer : layers) {
      delete layer;
    }
  }
  Matrix forward(const Matrix &input) {
    Matrix place_h = input;
    for (Linear *layer : layers) {
      Matrix output = layer->forward(place_h);
      place_h = output;
    }
    return place_h;
  }
  void backward(const Matrix &true_prob, const Matrix &pred_prob) {
    Matrix p_grad_output = sub(pred_prob, true_prob);
    for (int i = layers.size() - 1; i >= 0; i--) {

      Matrix grad_input = layers[i]->backward(p_grad_output);
      if (i > 0) {
        p_grad_output = hammard_product(
            grad_input, d_relu_copy(layers[i - 1]->get_outputs()));
      }
    }
  }

  void step(float lr) {
    for (Linear *layer : layers) {
      layer->get_weights() = sub(layer->get_weights(),
                                 layer->get_grad_weights().scale_inplace(lr));
      layer->get_biases() =
          sub(layer->get_biases(), layer->get_grad_biases().scale_inplace(lr));
    }
  }

private:
  std::vector<Linear *> layers;
};

Matrix read_data(std::string file, int rows, int cols) {
  FILE *f = fopen(file.c_str(), "rb");
  if (!f) {
    throw std::runtime_error("Could not open file");
  }
  int numel = rows * cols;
  float *data = new float[numel];
  Matrix dataset(rows, cols);
  size_t read_count = fread(data, sizeof(float), numel, f);
  fclose(f);
  if (read_count != numel) {
    delete[] data;
    throw std::runtime_error(
        "FILE DOES NOT HAVE THE EXPECTED NUMBER OF DATA ENTRIES");
  }
  dataset.copy_raw_array_and_delete(data);
  return dataset;
}
Matrix label_processing(std::string file, int rows) {
  Matrix labels = read_data(file, rows, 1);
  Matrix dataset_vector(rows, 10);
  dataset_vector.zero();
  for (int i = 0; i < rows; i++) {
    int cls = static_cast<int>(labels(i, 0));
    dataset_vector(i, cls) = 1.0f;
  }
  return dataset_vector;
}
float cross_entropy_loss(Matrix true_prob, Matrix pred_prob) {
  constexpr float EPS = 1e-8f;
  float sum = 0;
  for (int i = 0; i < pred_prob.rows(); ++i) {
    for (int j = 0; j < pred_prob.cols(); ++j) {
      float p = std::max(EPS, pred_prob(i, j));
      sum += true_prob(i, j) * log(p);
    }
  }
  return -sum / pred_prob.rows();
}

void training(model &a, int batch_size) {
  int steps = 50000 / batch_size;
  Matrix dataset = read_data("train_images.mat", 50000, 3072);
  Matrix labels = label_processing("train_labels.mat", 50000);
  Matrix batch_labels(batch_size, 10);
  for (int i = 0; i < 3; i++) {
    for (int i = 0; i < steps; i++) {
      Matrix inputs = dataset.slice_rows(i * batch_size, batch_size);
      batch_labels = labels.slice_rows(i * batch_size, batch_size);
      Matrix output = a.forward(inputs);
      softmax(output);

      a.backward(batch_labels, output);
      a.step(1e-4);
      std::cout << cross_entropy_loss(batch_labels, output) << std::endl;
    }
  }
}
int main() {
  Linear *a_ptr = new Linear(3072, 512, 0);
  Linear *b_ptr = new Linear(512, 10, -1);
  model model = {a_ptr, b_ptr};
  training(model, 100);

  return 0;
}
