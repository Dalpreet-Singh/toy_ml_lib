#include "Arena.hpp"

#include <cstdint>
Arena::Arena(size_t s) : size(s), buffer(::operator new(s)) {}

void *Arena::allocate(size_t bytes, size_t alignment) {
  assert((alignment & (alignment - 1)) == 0);
  std::byte *pl = static_cast<std::byte *>(buffer);
  std::byte *data = pl + total_bytes;

  uintptr_t p = reinterpret_cast<uintptr_t>(data);
  uintptr_t aligned = (p + (alignment - 1)) & ~(alignment - 1);
  std::byte *ptr = reinterpret_cast<std::byte *>(aligned);

  size_t padding = ptr - data;

  if (total_bytes + padding + bytes > size)
    return nullptr;

  total_bytes += padding + bytes;
  return ptr;
}
void Arena::reset() { total_bytes = 0; }
Arena::~Arena() {
  ::operator delete(buffer);
}

;
