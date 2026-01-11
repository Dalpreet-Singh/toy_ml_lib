

#include <cassert>
#include <cstddef>

#include <cstddef>

class Arena {
public:
  Arena(size_t s);

  void *allocate(size_t bytes, size_t alignment);
  void reset();
  ~Arena();

private:
  size_t size;
  void *buffer;
  size_t total_bytes = 0;
};
