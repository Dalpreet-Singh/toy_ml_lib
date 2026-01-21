
#include "model.hpp"
int main() {
  Linear *a = new Linear(3072, 512, 0);
  Linear *b = new Linear(512, 10, -1);
  model model{a, b};
  training(model, "train_images.mat", "train_labels.mat", 100, 50000, 3072, 5);

  return 0;
}
