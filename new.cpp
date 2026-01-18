#include "Arena.hpp"
#include "Matrix.hpp"
#include <cstdio>
#include <initializer_list>
#include <stdexcept>
#include <vector>
class Linear {
public:
  Linear(int in_features, int out_features, int od)
      : weights(out_features, in_features), inputs(Matrix(0, 0)),
        bias(1, out_features), order(od), grad_weights(0, 0), grad_biases(0, 0),
        outputs(Matrix(0, 0)) {}

  Matrix forward(Matrix input) {
    inputs = input;

    // input shape (b,i)
    // weight shape (o,i)
    // bias shape (o)

    Matrix output = broadcast_add((matmul(input, weights.T_C())), bias);
    outputs = output;

    return output;
  }
  Matrix backward(Matrix grad_output) {

    // inputs(b,i)
    grad_biases = grad_output.sum_dim_0();
    grad_weights = matmul(grad_output.T_C(), inputs);
    return matmul(grad_output, weights);
  }
  int get_order() { return order; }
  const Matrix &get_outputs() const { return outputs; }
  Matrix &get_weights() { return weights; }
  Matrix &get_grad_weights() { return grad_weights; }
  Matrix &get_grad_biases() { return grad_biases; }
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
  Matrix forward(Matrix input) {
    Matrix place_h = input;
    for (Linear *layer : layers) {
      Matrix output = layer->forward(place_h);
      place_h = output;
    }
    return place_h;
  }
  void backward(Matrix true_prob, Matrix pred_prob) {
    Matrix p_grad_output = sub(pred_prob, true_prob);
    for (int i = layers.size() - 1; i >= 0; i--) {

      Matrix grad_input = layers[i]->backward(p_grad_output);
      if (i > 0) {
        p_grad_output =
            hammard_product(grad_input, d_relu(layers[i - 1]->get_outputs()));
      }
    }
  }

  void step(float lr) {
    for (Linear *layer : layers) {
      layer->get_weights() =
          sub(layer->get_weights(), layer->get_grad_weights().scale(lr));
      layer->get_biases() =
          sub(layer->get_biases(), layer->get_grad_biases().scale(lr));
    }
  }

private:
  std::vector<Linear *> layers;
};

Matrix read_images(std::string file, int rows, int cols) {
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
void training_loop(model model, int batch_size) {}

int main() {

  Matrix a(5, 3);
  fill(a, 5);
  Matrix b(5, 3);
  fill(b, 6);
  b.T();
  Matrix out = matmul(a, b);

  out.print();

  return 0;
}
