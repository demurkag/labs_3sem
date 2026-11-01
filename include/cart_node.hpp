#pragma once

/** @file cart_node.hpp
 * @brief Узел дерева CART и локальная проверка его формы.
 */

#include <cstddef>
#include <memory>
#include <vector>
#include <cmath>

/** @brief Лист с ответом или внутренний узел с двумя потомками.
 * Потомки принадлежат узлу через unique_ptr; корень принадлежит CartTree.
 * Для регрессии лист хранит value, для классификации — probabilities.
 */
struct Node
{
    bool is_leaf = true; ///< Признак листа без потомков.

    std::size_t feature = 0; ///< Столбец разбиения внутреннего узла.
    double threshold = 0.0; ///< Порог: значения <= идут в left, остальные в right.

    double value = 0.0; ///< Среднее откликов регрессионного листа.
    std::vector<double> probabilities; ///< Частоты классов листа в порядке CartTree::classes().

    std::unique_ptr<Node> left; ///< Единоличное владение левым потомком.
    std::unique_ptr<Node> right; ///< Единоличное владение правым потомком.

    /** @brief Проверяет локальную форму, не обходя потомков.
     * @return Для листа — отсутствие потомков; для внутреннего узла — наличие
     * обоих потомков и конечность порога. Ответы и суммы вероятностей не проверяются.
     */
    bool is_valid() const
    {
        if (is_leaf)
        {
            return left == nullptr && right == nullptr;
        }

        return left != nullptr &&
               right != nullptr &&
               std::isfinite(threshold);
    }
};
