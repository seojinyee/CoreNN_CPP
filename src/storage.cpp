#include "storage.hpp"

// A simple wrapper for dynamic memory allocation
// Represents a 1D contiguous block of memory to store tensor data.
Storage::Storage(size_t size) : size_(size), data_(std::make_unique<float[]>(size)) {}
