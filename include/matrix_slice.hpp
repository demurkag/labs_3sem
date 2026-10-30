#pragma once

#include <cstddef>
#include <span>
#include <stdexcept>

enum class SliceAxis
{
    Row,
    Column
};

template <class T>
class MatrixSlice
{
private:
    T *data_ = nullptr;
    std::size_t size_ = 0;
    std::size_t stride_ = 1;
    SliceAxis axis_ = SliceAxis::Row;

public:
    MatrixSlice() noexcept = default;

    MatrixSlice(T *data, std::size_t size,
                std::size_t stride, SliceAxis axis)
        : data_(data),
          size_(size),
          stride_(stride),
          axis_(axis)
    {
        if (size == 0)
        {
            data_ = nullptr;
        }
        else
        {
            if (data == nullptr)
            {
                throw std::invalid_argument(
                    "Указатель не может быть nullptr для непустого среза");
            }

            if (stride == 0)
            {
                throw std::invalid_argument(
                    "Шаг непустого среза должен быть положительным");
            }
        }
    }

    std::size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }
    std::size_t stride() const noexcept { return stride_; }
    SliceAxis axis() const noexcept { return axis_; }
    bool is_contiguous() const noexcept { return size_ <= 1 || stride_ == 1; } // непрерывный
    T *data() const noexcept { return data_; }

    T &operator()(std::size_t index) const noexcept;
    T &at(std::size_t index) const;
    std::span<T> operator*() const;
    std::span<T> as_span() const; // та же проверка, что у operator*

    MatrixSlice<const double> as_const() const noexcept;
};