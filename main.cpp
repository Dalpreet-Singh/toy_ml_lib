
#include "model.hpp"
int main() {
  Linear *a = new Linear(3072, 512, 0);
  Linear *b = new Linear(512, 10, -1);
  model model{a, b};
  training(model, "train_images.bin", "train_labels.bin", 100, 50000, 3072, 5);
  eval(model, "test_images.bin", "test_labels.bin", 100, 10000, 3072);

  return 0;
}
