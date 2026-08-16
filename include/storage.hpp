#pragma once
#include <memory>
#include <cstddef>

// Manages the raw physical memory block for Tensors
class Storage{
private:
    std::unique_ptr<float[]> data_; // Smart pointer to the 1D float array
    size_t size_;                   // Total number of elements allocated

public:
    // Allocates a continuous block of memory of the given size
    explicit Storage(size_t size);
    ~Storage() = default;

    // Prevent accidental copying of raw memory (use shared_ptr for views instead)
    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;

    // Raw data accessors
    float* data() { return data_.get(); }
    const float* data() const { return data_.get(); }
    
    // Metadata accessor
    size_t size() const { return size_; }
};
