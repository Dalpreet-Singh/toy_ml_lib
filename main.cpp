
#include "model.hpp"
int main() {
  Linear a = Linear(3072, 512, 0);
  Linear b = (Linear(512, 10, -1));

  Linear *a_ptr = &a;
  Linear *b_ptr = &b;
  model model{a_ptr, b_ptr};
  training(model, "data/train_images.bin", "data/train_labels.bin", 100, 50000,
           3072, 5);
  eval(model, "data/test_images.bin", "data/test_labels.bin", 100, 10000, 3072);

  return 0;
}
