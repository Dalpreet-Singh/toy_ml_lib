

#include "utility.hpp"
[[noreturn]] void t_error(std::string error_code,
                          const std::source_location location) {
  std::cerr << "Error: " << error_code << "\n"
            << "File: " << location.file_name() << "\n"
            << "Line: " << location.line() << "\n"
            << "Function: " << location.function_name() << std::endl;
  std::exit(EXIT_FAILURE);
}
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
