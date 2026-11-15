#pragma once

/** @file random_forest_regressor.hpp
 * @brief Случайный лес регрессии на общем CART.
 */

#include "imodel.hpp"
#include "cart_regressor.hpp"
#include "random_forest_params.hpp"

/** @brief Владеет вектором CartRegressor и усредняет ответы деревьев.
 * Деревья создаются до запуска задач. Уникальные seed формируются общим
 * mt19937 в порядке номеров; каждая задача имеет собственные RNG и counts.
 * Обучение и получение ответов выполняются через forest_detail::run_tasks;
 * агрегация после join идёт последовательно в порядке деревьев.
 * Повторный fit начинает RNG заново. В одной среде ответы воспроизводимы
 * независимо от n_jobs; переносимость shuffle/distribution между библиотеками
 * C++ не гарантируется. Исходные X/y и counts после fit не хранятся по ссылке.
 * Дополнительная память пакетных ответов B*M double; суммирование O(B*M).
 * Численная устойчивость суммирования ответов отдельно не обеспечивается.
 */
class RandomForestRegressor : public IRegressor
{
private:
    RandomForestParams params_;
    std::vector<CartRegressor> trees_;

public:
    /** @brief Создаёт необученный лес с параметрами регрессии.
     * @param params Настройки; по умолчанию RandomForestParams{}, включая All.
     * @throws std::invalid_argument Нарушены независимые от данных ограничения RandomForestParams.
     */
    explicit RandomForestRegressor(
        RandomForestParams params = {});

    /** @brief Заново строит ансамбль, сначала удаляя прежние деревья.
     * @param X Конечная плотная матрица n x d, n >= 1, d >= 1; оба layout.
     * @param y Конечные отклики, ровно n значений.
     * @throws std::invalid_argument Неверные размеры, неконечные данные,
     * max_samples > n или Count > d.
     * @throws std::runtime_error Неконечная оценка разбиения CART.
     * @throws std::overflow_error Неконечное среднее листа CART.
     * @throws std::logic_error Нарушена форма узла CART.
     * @note Исключения задач передаются после join. При неудаче лес необучен;
     * частичный ансамбль не публикуется. Bootstrap-копии представлены counts.
     */
    void fit(
        const Matrix &X,
        std::span<const double> y) override;

    /** @brief Возвращает среднее ответов всех деревьев без изменения модели.
     * @param X Конечная матрица M x d с обучающим порядком признаков.
     * @return M ответов; при M=0 пустой вектор с проверкой числа столбцов.
     * @throws std::logic_error Лес не обучен.
     * @throws std::invalid_argument Неверное число признаков или неконечная X.
     * @note Ошибки рабочих задач передаются вызывающему коду после join.
     */
    std::vector<double> predict(
        const Matrix &X) const override;

    /** @brief Возвращает копии статистик в порядке номеров деревьев.
     * @return n_estimators записей; доступ к самим деревьям не предоставляется.
     * Стоимость — обход всех узлов ансамбля.
     * @throws std::logic_error Лес не обучен, в том числе после ошибки fit.
     */
    std::vector<TreeStats> tree_stats() const;
};
