#pragma once
#include "Matrix.hpp"
#include <cstddef>
#include <source_location>

#include <iostream>
[[noreturn]] void
t_error(std::string error_code,
        const std::source_location location = std::source_location::current());
Matrix read_data(std::string file, int rows, int cols);
Matrix label_processing(std::string file, int rows);
float cross_entropy_loss(Matrix true_prob, Matrix pred_prob);
