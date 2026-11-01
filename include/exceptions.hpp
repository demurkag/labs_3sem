#pragma once

/** @file exceptions.hpp
 * @brief Исключение преобразования неплотного среза в span.
 */

#include <stdexcept>

/** @brief Запрошен span для нескольких элементов среза с шагом не 1.
 * Сам срез допустим и остаётся доступен через поэлементные операции.
 * Конструкторы сообщений наследуются от std::logic_error.
 */
class NonContiguousSliceError : public std::logic_error
{
public:
    using std::logic_error::logic_error;
};
