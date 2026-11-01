#include "matrix.hpp"

#include <fstream>
#include <stdexcept>
#include <string>

std::size_t Matrix::index(std::size_t row, std::size_t col) const
{
    if (layout_ == Layout::RowMajor)
    {
        return row * cols_ + col;
    }

    return col * rows_ + row;
}

double &Matrix::operator()(std::size_t row, std::size_t col) &
{
    if (layout_ == Layout::RowMajor)
    {
        return data_[row * cols_ + col];
    }
    return data_[col * rows_ + row];
}

const double &Matrix::operator()(std::size_t row, std::size_t col) const &
{
    if (layout_ == Layout::RowMajor)
    {
        return data_[row * cols_ + col];
    }
    return data_[col * rows_ + row];
}

MatrixSlice<double> Matrix::operator()(std::size_t row, std::nullopt_t) &
{
    if (row >= rows_)
    {
        throw std::out_of_range("Индекс строки вне диапазона");
    }

    if (cols_ == 0)
    {
        return MatrixSlice<double>(nullptr, 0, 1, SliceAxis::Row);
    }

    std::size_t stride = (layout_ == Layout::RowMajor) ? 1 : rows_;

    return MatrixSlice<double>(&data_[index(row, 0)], cols_, stride, SliceAxis::Row);
}

MatrixSlice<const double> Matrix::operator()(std::size_t row, std::nullopt_t) const &
{
    if (row >= rows_)
    {
        throw std::out_of_range("Индекс строки вне диапазона");
    }

    if (cols_ == 0)
    {
        return MatrixSlice<const double>(nullptr, 0, 1, SliceAxis::Row);
    }

    std::size_t stride = (layout_ == Layout::RowMajor) ? 1 : rows_;

    return MatrixSlice<const double>(&data_[index(row, 0)], cols_, stride, SliceAxis::Row);
}

MatrixSlice<double> Matrix::operator()(std::nullopt_t, std::size_t col) &
{
    if (col >= cols_)
    {
        throw std::out_of_range("Индекс столбца вне диапазона");
    }

    if (rows_ == 0)
    {
        return MatrixSlice<double>(nullptr, 0, 1, SliceAxis::Column);
    }

    std::size_t stride = (layout_ == Layout::ColumnMajor) ? 1 : cols_;

    return MatrixSlice<double>(&data_[index(0, col)], rows_, stride, SliceAxis::Column);
}

MatrixSlice<const double> Matrix::operator()(std::nullopt_t, std::size_t col) const &
{
    if (col >= cols_)
    {
        throw std::out_of_range("Индекс столбца вне диапазона");
    }

    if (rows_ == 0)
    {
        return MatrixSlice<const double>(nullptr, 0, 1, SliceAxis::Column);
    }

    std::size_t stride = (layout_ == Layout::ColumnMajor) ? 1 : cols_;

    return MatrixSlice<const double>(&data_[index(0, col)], rows_, stride, SliceAxis::Column);
}

std::span<double> Matrix::operator[](std::size_t row) &
{
    if (layout_ != Layout::RowMajor)
    {
        throw std::runtime_error(
            "Matrix::operator[]: доступен только для RowMajor layout");
    }

    return {data_.data() + row * cols_, cols_};
}

std::span<const double> Matrix::operator[](std::size_t row) const &
{
    if (layout_ != Layout::RowMajor)
    {
        throw std::runtime_error(
            "Matrix::operator[]: доступен только для RowMajor layout");
    }

    return {data_.data() + row * cols_, cols_};
}

double &Matrix::at(std::size_t row, std::size_t col) &
{
    if (row >= rows_ || col >= cols_)
    {
        throw std::out_of_range(
            "Matrix::at: индекс (" +
            std::to_string(row) + ", " +
            std::to_string(col) +
            ") вне диапазона [" +
            std::to_string(rows_) + " x " +
            std::to_string(cols_) + "]");
    }

    return data_[index(row, col)];
}

const double &Matrix::at(std::size_t row, std::size_t col) const &
{
    if (row >= rows_ || col >= cols_)
    {
        throw std::out_of_range(
            "Matrix::at: индекс (" +
            std::to_string(row) + ", " +
            std::to_string(col) +
            ") вне диапазона [" +
            std::to_string(rows_) + " x " +
            std::to_string(cols_) + "]");
    }

    return data_[index(row, col)];
}

void Matrix::save(const std::filesystem::path &path) const
{
    std::ofstream file(path, std::ios::binary); // запись без каких-либо преобразований

    if (!file.is_open())
    {
        throw std::runtime_error("Не удалось открыть файл для записи: " + path.string());
    }

    file.write(reinterpret_cast<const char *>(&rows_), sizeof(rows_));
    file.write(reinterpret_cast<const char *>(&cols_), sizeof(cols_));
    file.write(reinterpret_cast<const char *>(&layout_), sizeof(layout_));
    file.write(reinterpret_cast<const char *>(data_.data()), data_.size() * sizeof(double));

    if (!file)
    {
        throw std::runtime_error("Ошибка записи матрицы в файл: " + path.string());
    }
}

Matrix Matrix::load(const std::filesystem::path &path)
{
    std::ifstream file(path, std::ios::binary);

    if (!file.is_open())
    {
        throw std::runtime_error("Не удалось открыть файл для чтения: " + path.string());
    }

    std::size_t rows = 0;
    std::size_t cols = 0;
    Layout layout = Layout::RowMajor;

    file.read(reinterpret_cast<char *>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char *>(&cols), sizeof(cols));
    file.read(reinterpret_cast<char *>(&layout), sizeof(layout));

    if (!file)
    {
        throw std::runtime_error("Повреждённый или неполный заголовок матрицы: " + path.string());
    }

    Matrix matrix(rows, cols, layout);

    auto &buffer = matrix.data();
    file.read(reinterpret_cast<char *>(buffer.data()), buffer.size() * sizeof(double));

    if (!file)
    {
        throw std::runtime_error("Ошибка чтения данных матрицы из файла: " + path.string());
    }

    return matrix;
}