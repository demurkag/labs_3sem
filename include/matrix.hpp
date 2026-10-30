#pragma once

/** @file matrix.hpp
 * @brief Плотная матрица double, доступ к элементам и бинарное хранение.
 */

#include "enums.hpp"
#include "matrix_slice.hpp"

#include <string>
#include <vector>
#include <filesystem>
#include <span>
#include <optional>

/**
 * @brief Плотная матрица double с порядком хранения RowMajor или ColumnMajor.
 * Представления строк и ссылки действуют, пока буфер не перераспределён и
 * объект не уничтожен. Размеры буфера нельзя менять через data(): это нарушает
 * соответствие rows(), cols() и физического хранилища.
 */
class Matrix
{
private:
    std::vector<double> data_;

    std::size_t rows_ = 0;
    std::size_t cols_ = 0;

    Layout layout_ = Layout::RowMajor;

    std::size_t index(std::size_t row, std::size_t col) const;

public:
    /** @brief Создаёт пустую матрицу 0 × 0 с порядком RowMajor. */
    Matrix() = default;

    /**
     * @brief Создаёт матрицу заданного размера, заполненную нулями.
     *
     * @param rows Число строк; ноль допустим.
     * @param cols Число столбцов; ноль допустим.
     * @param layout Порядок хранения элементов.
     *
     * @pre Произведение rows * cols не должно переполнять std::size_t.
     * @note Переполнение произведения размеров не проверяется.
     */
    Matrix(std::size_t rows, std::size_t cols, Layout layout)
        : data_(rows * cols, 0.0),
          rows_(rows),
          cols_(cols),
          layout_(layout) {}

    double &operator()(std::size_t row, std::size_t col) &;
    const double &operator()(std::size_t row, std::size_t col) const &;

    double &operator()(std::size_t row, std::size_t col) && = delete;
    const double &operator()(std::size_t row, std::size_t col) const && = delete;

    MatrixSlice<double> operator()(std::size_t row, std::nullopt_t) &;
    MatrixSlice<const double> operator()(std::size_t row, std::nullopt_t) const &;

    MatrixSlice<double> operator()(std::size_t row, std::nullopt_t) && = delete;
    MatrixSlice<const double> operator()(std::size_t row, std::nullopt_t) const && = delete;

    MatrixSlice<double> operator()(std::nullopt_t, std::size_t col) &;
    MatrixSlice<const double> operator()(std::nullopt_t, std::size_t col) const &;

    MatrixSlice<double> operator()(std::nullopt_t, std::size_t col) && = delete;
    MatrixSlice<const double> operator()(std::nullopt_t, std::size_t col) const && = delete;

    /**
     * @brief Возвращает изменяемое представление строки матрицы RowMajor.
     *
     * @param row Индекс строки, начиная с нуля.
     * @return std::span из cols() элементов без копирования.
     *
     * @pre Матрица имеет порядок хранения RowMajor.
     * @pre row < rows(); индекс строки не проверяется.
     *
     * @throws std::runtime_error Если порядок хранения не RowMajor.
     */
    std::span<double> operator[](std::size_t row);

    /**
     * @brief Возвращает представление строки матрицы RowMajor только для чтения.
     *
     * @param row Индекс строки, начиная с нуля.
     * @return Константное представление std::span из cols() элементов без копирования.
     *
     * @pre Матрица имеет порядок хранения RowMajor.
     * @pre row < rows(); индекс строки не проверяется.
     *
     * @throws std::runtime_error Если порядок хранения не RowMajor.
     */
    std::span<const double> operator[](std::size_t row) const;

    /**
     * @brief Предоставляет изменяемый элемент с проверкой границ для обоих Layout.
     * @param row Индекс строки с нуля.
     * @param col Индекс столбца с нуля.
     * @return Ссылка на элемент.
     * @throws std::out_of_range Если row >= rows() или col >= cols().
     */
    double &at(std::size_t row, std::size_t col);

    /**
     * @brief Читает элемент с проверкой границ для обоих Layout.
     * @param row Индекс строки с нуля.
     * @param col Индекс столбца с нуля.
     * @return Константная ссылка на элемент.
     * @throws std::out_of_range Если row >= rows() или col >= cols().
     */
    const double &at(std::size_t row, std::size_t col) const;

    /** @brief Возвращает число строк.
     * @return Число строк матрицы.
     */
    std::size_t rows() const { return rows_; }

    /** @brief Возвращает число столбцов.
     * @return Число столбцов матрицы.
     */
    std::size_t cols() const { return cols_; }

    /**
     * @brief Предоставляет изменяемый доступ к физическому буферу.
     * @return Ссылка на вектор элементов.
     * @warning Нельзя изменять размер буфера, поскольку это нарушает
     * соответствие данных размерам матрицы.
     */
    std::vector<double> &data() { return data_; }

    /**
     * @brief Предоставляет доступ к физическому буферу для чтения.
     * @return Константная ссылка на вектор элементов.
     */
    const std::vector<double> &data() const { return data_; }

    /**
     * @brief Сохраняет матрицу в бинарный файл.
     *
     * Записывает размеры матрицы, порядок хранения Layout
     * и все элементы типа double в бинарном формате платформы.
     *
     * @param path Путь к файлу; существующий файл перезаписывается.
     * @throws std::runtime_error При ошибке открытия или записи файла.
     *
     * @note Родительские каталоги не создаются.
     */
    void save(const std::filesystem::path &path) const;

    /**
     * @brief Загружает матрицу из бинарного файла.
     *
     * Считывает размеры и порядок хранения из заголовка,
     * выделяет память и загружает элементы матрицы.
     *
     * @param path Путь к файлу в формате save() на совместимой платформе.
     * @return Загруженная матрица с сохранёнными размерами и Layout.
     *
     * @pre Файл должен содержать корректные размеры и значение Layout.
     *      Произведение размеров не должно переполнять std::size_t.
     *
     * @throws std::runtime_error Если файл не удалось открыть
     *         или заголовок либо данные прочитаны не полностью.
     *
     * @warning Метод не проверяет допустимость размеров и Layout.
     *          Не рекомендуется загружать файлы из недоверенных источников.
     *
     * @note Бинарный формат не содержит версии и контрольной суммы.
     */
    static Matrix load(const std::filesystem::path &path); // Named Constructor Idiom / несколько "конструкторов"
};
