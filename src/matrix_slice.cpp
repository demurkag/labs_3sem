#include "matrix_slice.hpp"
#include "exceptions.hpp"

#include <stdexcept>

template <class T>
T &MatrixSlice<T>::operator()(std::size_t index) const noexcept
{
    return data_[index * stride_];
}

template <class T>
T &MatrixSlice<T>::at(std::size_t index) const
{
    if (index >= size_)
    {
        throw std::out_of_range("Индекс среза вне диапазона");
    }
    return (*this)(index);
}

template <class T>
std::span<T> MatrixSlice<T>::operator*() const
{
    if (!is_contiguous())
    {
        throw NonContiguousSliceError(
            "Невозможно получить std::span из неплотного среза");
    }

    return std::span<T>(data_, size_);
}

template <class T>
std::span<T> MatrixSlice<T>::as_span() const
{
    return this->operator*();
}

template <class T>
MatrixSlice<const double> MatrixSlice<T>::as_const() const noexcept
{
    return MatrixSlice<const double>(data_, size_, stride_, axis_);
}

template class MatrixSlice<double>;
template class MatrixSlice<const double>;
