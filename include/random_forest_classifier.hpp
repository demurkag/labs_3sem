#pragma once

/** @file random_forest_classifier.hpp
 * @brief Случайный лес классификации с усреднением вероятностей.
 */

#include "imodel.hpp"
#include "cart_classifier.hpp"
#include "random_forest_params.hpp"

#include <span>
#include <vector>

/** @brief Владеет вектором CartClassifier с общим порядком исходных меток.
 * Каждый CART строит classes по всему y, даже если counts класса нулевые.
 * Отсутствующие в bootstrap классы имеют нулевую вероятность. Ансамбль
 * усредняет вероятности, затем выбирает метку; голоса деревьев не используются.
 * Уникальные seed назначаются до запуска задач. RNG и counts локальны,
 * модели и ячейки результатов подготовлены заранее. После join вероятности
 * суммируются в порядке деревьев. Повторный fit начинает RNG заново;
 * n_jobs не меняет ответы в одной среде, но межбиблиотечное совпадение RNG
 * не гарантируется. Модель не сохраняет невладеющие ссылки на X/y/counts.
 * Пакетные вероятности требуют дополнительно B*M*C double;
 * последовательная агрегация стоит O(B*M*C).
 */
class RandomForestClassifier : public IClassifier
{
private:
    RandomForestParams params_;
    std::vector<CartClassifier> trees_;

public:
    /** @brief Создаёт необученный лес; max_features=Sqrt, остальные настройки
     * соответствуют значениям по умолчанию RandomForestParams.
     */
    RandomForestClassifier();

    /** @brief Создаёт лес с переданными настройками без замены max_features.
     * @param params Настройки; RandomForestParams{} явно задаёт All.
     * @throws std::invalid_argument Нарушены независимые от данных ограничения RandomForestParams.
     */
    explicit RandomForestClassifier(
        RandomForestParams params);

    /** @brief Заново обучает ансамбль и глобальную схему классов.
     * @param X Конечная плотная матрица n x d, n >= 1, d >= 1; оба layout.
     * @param y Ровно n конечных исходных числовых меток; один класс допустим.
     * @throws std::invalid_argument Неверные размеры, неконечные данные,
     * max_samples > n или Count > d.
     * @throws std::runtime_error Неконечная оценка разбиения CART.
     * @throws std::logic_error Нарушена форма узла CART.
     * @note Прежний лес сбрасывается до проверки данных; при любой ошибке
     * модель необучена. Исключения задач передаются после join.
     */
    void fit(
        const Matrix &X,
        std::span<const double> y) override;

    /** @brief Выбирает метку по средним вероятностям predict_proba().
     * @param X Конечная матрица M x d с обучающим порядком столбцов.
     * @return M исходных меток; при ничьей меньшая метка; при M=0 пусто.
     * @throws std::logic_error Лес не обучен.
     * @throws std::invalid_argument Неверное число столбцов или неконечная X.
     */
    std::vector<double> predict(
        const Matrix &X) const override;

    /** @brief Усредняет вероятности деревьев без изменения обученного состояния.
     * @param X Конечная матрица M x d с обучающим порядком столбцов.
     * @return RowMajor-матрица M x C; столбец c соответствует classes()[c].
     * При M=0 возвращается 0 x C с проверкой числа столбцов.
     * @throws std::logic_error Лес не обучен.
     * @throws std::invalid_argument Неверное число столбцов или неконечная X.
     * @note Исключения рабочих задач передаются после join.
     */
    Matrix predict_proba(
        const Matrix &X) const override;

    /** @brief Возвращает уникальные исходные метки всего y_train по возрастанию.
     * @return Невладеющий span; действует до следующего fit или уничтожения
     * модели. Порядок не зависит от запросов и отсутствующих bootstrap-классов.
     * @throws std::logic_error Лес не обучен, в том числе после ошибки fit.
     */
    std::span<const double> classes() const override;

    /** @brief Возвращает копии статистик в порядке номеров деревьев.
     * @return n_estimators записей без доступа к самим деревьям;
     * стоимость — обход всех узлов ансамбля.
     * @throws std::logic_error Лес не обучен, в том числе после ошибки fit.
     */
    std::vector<TreeStats> tree_stats() const;
};
