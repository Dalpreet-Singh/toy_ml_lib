#include "Arena.hpp"
#include "Matrix.hpp"
#include <iostream>

int main() {

  Matrix a(5, 3);
  fill(a, 5);
  Matrix b(5, 3);
  fill(b, 6);
  b.T();
  Matrix out(5, 5);
  matmul(out, a, b);

  out.print();

  return 0;
}
