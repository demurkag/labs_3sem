#pragma once

/** @file forest_tasks.hpp
 * @brief Ограниченный запуск независимых задач ансамбля через std::jthread.
 */

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <exception>
#include <thread>
#include <vector>

/** @brief Внутренняя инфраструктура выполнения задач леса. */
namespace forest_detail
{
    /** @brief Выполняет номера задач с ограничением числа рабочих потоков.
     * @tparam Function Вызываемый объект с аргументом std::size_t.
     * @param task_count Число задач; ноль допустим.
     * @param n_jobs Положительный предел числа рабочих потоков.
     * @param task Независимая задача; общие входы только читаются, выходы
     * и ошибки принадлежат отдельным заранее подготовленным ячейкам.
     * @pre n_jobs >= 1; ссылки внутри task живут до возврата функции.
     * @note При min(n_jobs,task_count)=1 вызовы последовательны, исключение
     * передаётся сразу. Иначе рабочий w выполняет w,w+p,...; исключения
     * сохраняются в exception_ptr, оставшиеся задачи продолжаются. После join
     * всех потоков повторно выбрасывается ошибка с наименьшим номером задачи.
     * Тип и содержимое исключения сохраняются. Агрегация не выполняется здесь.
     */
    template <class Function>
    void run_tasks(std::size_t task_count, std::size_t n_jobs, const Function &task)
    {
        assert(n_jobs > 0);
        if (task_count == 0)
            return;

        const std::size_t p = std::min(n_jobs, task_count);
        if (p == 1)
        {
            for (std::size_t i = 0; i < task_count; ++i)
                task(i);
            return;
        }

        std::vector<std::exception_ptr> errors(task_count);
        std::vector<std::jthread> workers;
        workers.reserve(p);

        for (std::size_t w = 0; w < p; ++w)
        {
            workers.emplace_back([&, w]
                                 {
                for (std::size_t i = w; i < task_count; i += p)
                {
                    try
                    {
                        task(i);
                    }
                    catch (...)
                    {
                        errors[i] = std::current_exception();
                    }
                } });
        }
        for (auto &worker : workers)
            worker.join();

        for (const auto &error : errors)
            if (error)
                std::rethrow_exception(error);
    }
} // namespace forest_detail
