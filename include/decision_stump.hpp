#pragma once

/** @file decision_stump.hpp
 * @brief Публичный интерфейс решающего пня.
 */

#include "imodel.hpp"
#include "enums.hpp"

#include <cstddef>
#include <span>
#include <vector>
#include <stdexcept>

/**
 * @brief Модель с одним разбиением по признаку и порогу, реализующая IModel.
 * До успешного fit() предсказание недоступно. Если допустимого разбиения нет,
 * обученная модель предсказывает одну константу. Геттеры сами не проверяют
 * состояние обучения; индекс и порог значимы только при has_split() == true.
 */
class DecisionStump : public IModel
{
private:
    std::size_t feature_index_ = 0; // признак
    double threshold_ = 0.0;        // порог

    double left_value_ = 0.0;
    double right_value_ = 0.0;

    SplitCriterion criterion_;
    std::size_t min_samples_leaf_;

    std::size_t n_features_ = 0;
    bool is_fitted_ = false;
    bool has_split_ = false;

public:
    /**
     * @brief Создаёт необученный пень.
     * @param criterion Variance для регрессии или Gini для классификации.
     * @param min_samples_leaf Минимальное число объектов с каждой стороны разбиения.
     * @throws std::invalid_argument Если критерий неизвестен или min_samples_leaf равен нулю.
     * @note Слишком большое положительное значение допустимо, но может запретить разбиение.
     */
    DecisionStump(SplitCriterion criterion, std::size_t min_samples_leaf)
        : criterion_(criterion),
          min_samples_leaf_(min_samples_leaf)
    {
        if (criterion_ != SplitCriterion::Variance &&
            criterion_ != SplitCriterion::Gini)
            throw std::invalid_argument("Неизвестный критерий разбиения");

        if (min_samples_leaf_ == 0)
            throw std::invalid_argument("min_samples_leaf должен быть больше нуля");
    }

    /**
     * @brief Обучает пень, заменяя результат предыдущего успешного обучения.
     * @param features Непустая матрица с хотя бы одним признаком.
     * @param target Отклик для каждой строки; при Gini числа служат метками классов.
     * @throws std::invalid_argument Нет строк/признаков, длина target неверна
     * или features/target содержат NaN либо бесконечность.
     * @throws std::runtime_error SplitFinder вычислил неконечную оценку.
     * @note Без разбиения оба листа содержат среднее (Variance) или наиболее частый
     * класс (Gini); при равенстве частот выбирается меньшая числовая метка.
     * При строго одинаковой оценке разбиений сохраняется первое рассмотренное:
     * меньший индекс признака, затем первая граница в порядке возрастания значений.
     * При исключении пень необучен; прежние значения геттеров могут сохраняться.
     * Поиск выполняется общим SplitFinder. Постоянный отклик Variance допускает
     * разбиение; для Gini единственный класс сразу даёт константный ответ.
     * Переполнение сумм среднего листа отдельно не проверяется.
     */
    void fit(const Matrix &features, std::span<const double> target) override;

    /**
     * @brief Предсказывает по правилу: значение не больше порога относится к левому листу.
     * @param features Матрица с тем же числом и порядком признаков, что при fit().
     * @return Предсказания в порядке строк; для нуля строк — пустой вектор.
     * @pre Порядок столбцов вызывающая сторона сохраняет сама.
     * @throws std::logic_error Если успешного обучения ещё не было.
     * @throws std::invalid_argument Число признаков отличается от обучающего
     * или признаки содержат NaN/бесконечность.
     * @note При отсутствии разбиения результат постоянен для всех строк.
     */
    std::vector<double> predict(const Matrix &features) const override;

    /** @brief Возвращает индекс выбранного признака.
     * @return Индекс с нуля; значим только после успешного fit() при has_split().
     */
    std::size_t split_feature() const { return feature_index_; }

    /** @brief Возвращает порог выбранного разбиения.
     * @return Порог; значим только после успешного fit() при has_split().
     */
    double split_threshold() const { return threshold_; }

    /** @brief Сообщает, найдено ли разбиение.
     * @return false до обучения и при обучении без допустимого разбиения.
     */
    bool has_split() const { return has_split_; }

    /** @brief Возвращает предсказание левого листа.
     * @return Среднее или метка класса; до обучения начальное значение 0, не прогноз.
     */
    double left_value() const { return left_value_; }

    /** @brief Возвращает предсказание правого листа.
     * @return Среднее или метка класса; без разбиения совпадает с левым листом после fit().
     */
    double right_value() const { return right_value_; }
};
