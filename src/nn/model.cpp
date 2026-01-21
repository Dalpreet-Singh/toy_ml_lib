

#include "model.hpp"
#include <chrono>
#include <cmath>
#include <initializer_list>
#include <iostream>
#include <random>
#include <vector>
Linear::Linear(int in_features, int out_features, int od)
    : weights(out_features, in_features), inputs(Matrix(0, 0)),
      bias(1, out_features), order(od), grad_weights(0, 0), grad_biases(0, 0),
      outputs(Matrix(0, 0)) {
  if (in_features <= 0 || out_features <= 0) {
    t_error("in and out features must be greater than 0");
  } else if (od < -1) {
    t_error("order cannot be any negative number except -1 which symbolizes "
            "the last layer");
  }
}

Matrix Linear::forward(const Matrix &input) {
  if (input.cols() != weights.cols()) {
    t_error("Inputs cols does not match up with weight cols before transpose");
  }
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
void Linear::init() {
  std::random_device rd;
  std::mt19937 gen(rd());
  float in_fan = static_cast<float>(weights.cols());
  float gain = (order != -1) ? 2 : 1;
  std::uniform_real_distribution<float> dist(-std::sqrt(gain * 3.0f / in_fan),
                                             std::sqrt(gain * 3.0f / in_fan));

  for (int i = 0; i < weights.rows(); i++) {
    for (int j = 0; j < weights.cols(); j++) {

      weights(i, j) = dist(gen);
    }
  }
  bias.zero();
}
Matrix Linear::backward(const Matrix &grad_output) {
  if (grad_output.rows() != inputs.rows() ||
      grad_output.cols() != weights.rows()) {
    t_error("shape of grad output does not batch up with (batch_size, "
            "out_features)");
  }
  // inputs(b,i)
  grad_biases = grad_output.sum_dim_0_copy();
  grad_weights = matmul(grad_output.T_C(), inputs);
  return matmul(grad_output, weights);
}
int Linear::get_order() { return order; }
const Matrix &Linear::get_outputs() const { return outputs; }
Matrix &Linear::get_weights() { return weights; }
Matrix Linear::get_grad_weights() const { return grad_weights; }
Matrix Linear::get_grad_biases() const { return grad_biases; }
Matrix &Linear::get_biases() { return bias; }

model::model(std::initializer_list<Linear *> list) {
  if (list.size() <= 0) {
    t_error("you must have atleast 1 layer in the model");
  }
  for (Linear *layer : list) {

    layers.push_back(layer);
  }
}
model::~model() {
  for (Linear *layer : layers) {
    delete layer;
  }
}
Matrix model::forward(const Matrix &input) {
  Matrix place_h = input;
  for (Linear *layer : layers) {
    Matrix output = layer->forward(place_h);
    place_h = output;
  }
  return place_h;
}
void model::init() {
  for (Linear *layer : layers) {
    layer->init();
  }
}
void model::backward(const Matrix &true_prob, const Matrix &pred_prob) {
  if (true_prob.rows() != pred_prob.rows() ||
      true_prob.cols() != pred_prob.cols()) {
    t_error("Shape of predicted probalities and true probabilities does not "
            "match(required for subtraction to compute dL/dA)");
  }
  Matrix p_grad_output = sub(pred_prob, true_prob);
  for (int i = layers.size() - 1; i >= 0; i--) {

    Matrix grad_input = layers[i]->backward(p_grad_output);
    if (i > 0) {
      p_grad_output = hammard_product(
          grad_input, d_relu_copy(layers[i - 1]->get_outputs()));
    }
  }
}

void model::step(float lr) {
  for (Linear *layer : layers) {
    layer->get_weights() =
        sub(layer->get_weights(), layer->get_grad_weights().scale_inplace(lr));
    layer->get_biases() =
        sub(layer->get_biases(), layer->get_grad_biases().scale_inplace(lr));
  }
}

void training(model &a, std::string train_file, std::string train_label_file,
              int batch_size, int rows, int cols, int epochs) {
  auto start = std::chrono::steady_clock::now();

  a.init();
  int steps = rows / batch_size;
  Matrix dataset = read_data(train_file, rows, cols);
  Matrix labels = label_processing(train_label_file, rows);
  Matrix batch_labels(batch_size, 10);
  for (int i = 0; i < epochs; i++) {
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
  auto end = std::chrono::steady_clock::now();

  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  std::cout << "Time elapsed: " << duration.count() << " milliseconds"
            << std::endl;
}
