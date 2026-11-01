#pragma once

/** @file cart_tree.hpp
 * @brief Общее дерево, статистика и описания разбиений CART.
 */

#include "cart_node.hpp"
#include "split_finder.hpp"

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

/** @brief Размер и фактическая глубина обученного дерева. */
struct TreeStats
{
    std::size_t node_count = 0; ///< Все узлы, включая листья.
    std::size_t leaf_count = 0; ///< Число листьев.
    std::size_t depth = 0;      ///< Максимальное число рёбер от корня до листа; у одного листа 0.
};

/** @brief Описание внутреннего узла для вывода приложением. */
struct SplitInfo
{
    std::size_t depth = 0;   ///< Глубина узла, начиная с 0 у корня.
    std::size_t feature = 0; ///< Индекс столбца исходной матрицы признаков.
    double threshold = 0.0;  ///< Порог маршрутизации по правилу <= влево.
};

/** @brief Общее рекурсивное дерево для CartRegressor и CartClassifier.
 * Владеет корнем и через него потомками посредством unique_ptr. На каждом шаге
 * передаются индексы строк, без копирования подматрицы; разбиение ищет SplitFinder,
 * общий с DecisionStump. Лист создаётся при достижении max_depth, числе объектов
 * меньше min_samples_split, одинаковых откликах или отсутствии допустимого порога.
 * Отдельной проверки улучшения критерия нет; ничьи разрешает SplitFinder.
 * Регрессионный лист хранит среднее, классификационный — доли классов.
 * Стоимость поиска суммируется по внутренним узлам; обход одного пути O(depth).
 * Публичные predict оболочек дополнительно проверяют всю входную матрицу.
 * При вычислении среднего проверяется конечность результата.
 * Если среднее не может быть представлено конечным числом,
 * обучение завершается исключением.
 */
class CartTree
{
private:
    std::unique_ptr<Node> root_;

    SplitCriterion criterion_;

    std::size_t max_depth_;
    std::size_t min_samples_split_;
    std::size_t min_samples_leaf_;

    std::size_t n_features_ = 0;

    SplitFinder finder_;

    std::vector<double> classes_;

    std::unique_ptr<Node> build_tree(
        const Matrix &X,
        std::span<const double> y,
        const std::vector<std::size_t> &object_ids,
        const std::vector<std::size_t> &feature_ids,
        std::size_t depth);

    std::unique_ptr<Node> make_leaf(
        std::span<const double> y,
        const std::vector<std::size_t> &object_ids);

    void collect_stats(
        const Node *node,
        std::size_t depth,
        TreeStats &result) const;

    void collect_splits(
        const Node *node,
        std::size_t depth,
        std::size_t limit,
        std::vector<SplitInfo> &result) const;

public:
    /** @brief Создаёт необученное дерево.
     * @param criterion Variance для регрессии или Gini для классификации.
     * @param max_depth Максимум рёбер от корня; 0 означает только один лист.
     * @param min_samples_split Минимум объектов для попытки разбиения, не меньше 2.
     * @param min_samples_leaf Минимум объектов в каждой стороне разбиения, не меньше 1.
     * @throws std::invalid_argument Неизвестный критерий или недопустимые минимумы.
     */
    CartTree(
        SplitCriterion criterion,
        std::size_t max_depth,
        std::size_t min_samples_split,
        std::size_t min_samples_leaf);

    /** @brief Проверяет обучение, число столбцов и конечность всех признаков.
     * @param X Матрица предсказания; ноль строк допустим.
     * @throws std::logic_error Дерево не обучено.
     * @throws std::invalid_argument Число столбцов отличается или есть NaN/бесконечность.
     */
    void validate_predict_input(const Matrix &X) const;

    /** @brief Обучает дерево, сначала сбрасывая прежнее состояние.
     * @param X Непустая матрица с хотя бы одним столбцом, значения конечны.
     * @param y Конечные отклики всех строк X; при Gini — метки классов.
     * @throws std::invalid_argument Пустые размеры, неверная длина y или неконечный вход.
     * @throws std::runtime_error SplitFinder получил неконечную оценку.
     * @throws std::logic_error Построенный узел не прошёл локальную проверку формы.
     * @note Успех заменяет прежнее дерево. При исключении корня нет и is_fitted()
     * возвращает false; промежуточный список классов может остаться, но недоступен
     * через classes(). Данные X/y не сохраняются как невладеющие представления.
     */
    void fit(const Matrix &X, std::span<const double> y);

    /** @brief Находит лист по правилу X(row, feature) <= threshold влево.
     * @param X Матрица с обучающим числом и порядком столбцов.
     * @param row Индекс объекта в X.
     * @return Ссылка на лист до следующего fit() или уничтожения дерева.
     * @throws std::logic_error Дерево не обучено.
     * @throws std::invalid_argument Число столбцов отличается от обучающего.
     * @throws std::out_of_range row вне X.
     * @pre Значения X конечны; этот метод сам их не проверяет.
     */
    const Node &find_leaf(
        const Matrix &X,
        std::size_t row) const;

    /** @brief Проверяет наличие корня. @return true после успешного fit(). */
    bool is_fitted() const noexcept;

    /** @brief Возвращает классы обученного дерева.
     * @return Невладеющий отсортированный список меток Gini; для Variance пуст.
     * Действует до следующего fit() или уничтожения дерева.
     * @throws std::logic_error Дерево не обучено.
     */
    std::span<const double> classes() const;

    /** @brief Обходит дерево и подсчитывает его размер.
     * @return Число узлов, листьев и максимальная глубина; стоимость O(node_count).
     * @throws std::logic_error Дерево не обучено.
     */
    TreeStats stats() const;

    /** @brief Собирает внутренние узлы в порядке: узел, левое, правое поддерево.
     * @param limit Максимум записей; 0 возвращает пустой вектор.
     * @return Не более limit разбиений; у одного листа вектор пуст.
     * @throws std::logic_error Дерево не обучено.
     */
    std::vector<SplitInfo> first_splits(
        std::size_t limit) const;
};
