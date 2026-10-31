#pragma once

/** @file imodel.hpp
 * @brief Абстрактный интерфейс модели машинного обучения.
 *
 * Определяет общие методы обучения и предсказания.
 * Конкретные реализации определяют алгоритм обучения,
 * ограничения входных данных и возможные исключения.
 */

#include "matrix.hpp"
#include <vector>
#include <span>

/**
 * @brief Интерфейс модели, доступной через ссылку или указатель на IModel.
 * Конкретная реализация задаёт критерий обучения, предусловия и исключения.
 */
class IModel
{
public:
    /** @brief Позволяет безопасно уничтожать модель через указатель на интерфейс. */
    virtual ~IModel() = default;

    /**
     * @brief Обучает модель на признаках и соответствующих откликах.
     *
     * @param features Матрица: строки — объекты, столбцы — признаки.
     * @param target Невладеющее представление откликов,
     *               соответствующих строкам features.
     *
     * @pre Количество элементов target должно совпадать
     *      с количеством строк features.
     *
     * @note Проверки аргументов и состояние модели
     *       при ошибке зависят от конкретной реализации.
     */
    virtual void fit(
        const Matrix &features,
        std::span<const double> target) = 0;

    /**
     * @brief Возвращает предсказания обученной модели.
     *
     * @param features Матрица объектов для предсказания.
     * @return Вектор предсказаний, по одному значению на строку.
     *         Смысл значений определяется конкретной моделью.
     *
     * @pre Модель должна быть обучена перед вызовом predict().
     *
     * @note Требования к количеству признаков и исключения
     *       зависят от конкретной реализации.
     */
    virtual std::vector<double> predict(
        const Matrix &features) const = 0;
};

class IRegressor : public IModel
{
};

class IClassifier : public IModel
{
public:
    virtual Matrix predict_proba(const Matrix &X) const = 0;

    virtual std::span<const double> classes() const = 0;
};
