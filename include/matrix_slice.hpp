#pragma once

/** @file matrix_slice.hpp
 * @brief Невладеющие срезы строк и столбцов матрицы.
 */

#include <cstddef>
#include <span>
#include <stdexcept>

/** @brief Логическое направление среза независимо от физического layout. */
enum class SliceAxis
{
    Row,    ///< Строка: меняется индекс столбца.
    Column  ///< Столбец: меняется индекс строки.
};

/** @brief Невладеющий срез без копирования элементов.
 * @tparam T double для изменения данных или const double для чтения;
 * определения в .cpp явно инстанцированы для этих двух типов.
 * Элемент i расположен по адресу data() + i * stride(). Шаг измеряется
 * в элементах, не байтах. В RowMajor строки плотные, столбцы имеют шаг cols();
 * в ColumnMajor столбцы плотные, строки имеют шаг rows().
 * const у оболочки не запрещает изменение T=double; для чтения нужен const T.
 * Владелец буфера должен жить дольше среза и полученных ссылок/span;
 * перераспределение или замена буфера может сделать их недействительными.
 */
template <class T>
class MatrixSlice
{
private:
    T *data_ = nullptr;
    std::size_t size_ = 0;
    std::size_t stride_ = 1;
    SliceAxis axis_ = SliceAxis::Row;

public:
    /** @brief Создаёт пустой срез строки с шагом 1. */
    MatrixSlice() noexcept = default;

    /** @brief Создаёт срез над существующим буфером.
     * @param data Указатель на первый элемент; для size=0 сохраняется nullptr.
     * @param size Число логических элементов.
     * @param stride Шаг между элементами в единицах T.
     * @param axis Логическое направление среза.
     * @pre Буфер покрывает все адресуемые элементы; произведение индекса и шага не переполняется.
     * @throws std::invalid_argument Непустой срез имеет nullptr или нулевой шаг.
     */
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

    /** @brief Возвращает длину. @return Число элементов. */
    std::size_t size() const noexcept { return size_; }
    /** @brief Проверяет пустоту. @return true при size()==0. */
    bool empty() const noexcept { return size_ == 0; }
    /** @brief Возвращает физический шаг. @return Расстояние между элементами в единицах T. */
    std::size_t stride() const noexcept { return stride_; }
    /** @brief Возвращает направление. @return Row для строки, Column для столбца. */
    SliceAxis axis() const noexcept { return axis_; }
    /** @brief Проверяет возможность представления в виде span.
     * @return true для длины 0 или 1 либо шага 1.
     */
    bool is_contiguous() const noexcept { return size_ <= 1 || stride_ == 1; } // непрерывный
    /** @brief Возвращает начало буфера. @return Указатель с константностью T; у пустого среза nullptr. */
    T *data() const noexcept { return data_; }

    /** @brief Доступ с учётом шага без проверки границ.
     * @param index Индекс с нуля, меньше size().
     * @return Ссылка на элемент с константностью T.
     */
    T &operator()(std::size_t index) const noexcept;
    /** @brief Доступ с проверкой границ, включая неплотные срезы.
     * @param index Индекс с нуля.
     * @return Ссылка на элемент с константностью T.
     * @throws std::out_of_range index не меньше size().
     */
    T &at(std::size_t index) const;
    /** @brief Представляет непрерывные элементы без копирования.
     * @return span длины size() с константностью T; пустой срез допустим.
     * @throws NonContiguousSliceError Более одного элемента при шаге не 1.
     * Неплотный срез остаётся доступен через operator() и at().
     */
    std::span<T> operator*() const;
    /** @brief Именованный эквивалент operator*().
     * @return Невладеющий span с константностью T.
     * @throws NonContiguousSliceError Срез неплотный и содержит несколько элементов.
     */
    std::span<T> as_span() const; // та же проверка, что у operator*

    /** @brief Создаёт оболочку только для чтения тех же элементов.
     * @return Срез const double с теми же размером, шагом и направлением; без копирования данных.
     */
    MatrixSlice<const double> as_const() const noexcept;
};
