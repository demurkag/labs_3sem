#include "dataset.hpp"
#include "cart_classifier.hpp"
#include "cart_regressor.hpp"
#include "random_forest_classifier.hpp"
#include "random_forest_regressor.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#ifndef LAB4_BUILD_TYPE
#define LAB4_BUILD_TYPE "Unknown"
#endif

using Clock = std::chrono::steady_clock;

template <class Function>
static double measure_ms(Function action)
{
    const auto start = Clock::now();
    action();
    const auto stop = Clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

static void print_parameters(std::string_view features,
                             std::size_t jobs)
{
    std::cout << "\nCART: max_depth=3, min_samples_split=2, min_samples_leaf=1, max_features=All\n\n"
              << "Random Forest: n_estimators=100, max_depth=3, min_samples_split=2, "
              << "min_samples_leaf=1, bootstrap=true, max_samples=nullopt, max_features="
              << features << ", random_state=42, n_jobs=" << jobs << "\n\n";
}

static void print_stats(const TreeStats &stats)
{
    std::cout << "nodes=" << stats.node_count << ", leaves=" << stats.leaf_count
              << ", depth=" << stats.depth;
}

static void print_forest_stats(const std::vector<TreeStats> &stats)
{
    std::size_t nodes = 0, leaves = 0, depth_sum = 0, max_depth = 0;
    for (const auto &tree : stats)
    {
        nodes += tree.node_count;
        leaves += tree.leaf_count;
        depth_sum += tree.depth;
        max_depth = std::max(max_depth, tree.depth);
    }

    std::cout << "\nForest статистика: trees=" << stats.size() << ", nodes=" << nodes
              << ", leaves=" << leaves << ", mean_nodes=" << double(nodes) / stats.size()
              << ", mean_depth=" << double(depth_sum) / stats.size()
              << ", max_depth=" << max_depth << "\n\n";

    for (std::size_t i = 0; i < std::min<std::size_t>(5, stats.size()); ++i)
    {
        std::cout << "  tree " << i << ": ";
        print_stats(stats[i]);
        std::cout << '\n';
    }
}

static double accuracy(std::span<const double> predicted,
                       std::span<const double> truth)
{
    if (predicted.size() != truth.size() || truth.empty())
        throw std::runtime_error("Некорректные размеры для accuracy");

    std::size_t correct = 0;
    for (std::size_t i = 0; i < truth.size(); ++i)
        correct += predicted[i] == truth[i];

    return double(correct) / truth.size();
}

static void print_probabilities(const std::vector<std::string> &number_to_label,
                                std::span<const double> classes,
                                const Matrix &probabilities,
                                std::size_t row)
{
    for (std::size_t c = 0; c < classes.size(); ++c)
        std::cout << (c == 0 ? "" : ", ")
                  << number_to_label.at(static_cast<std::size_t>(classes[c]))
                  << '=' << probabilities(row, c);
    std::cout << '\n';
}

static double median(std::vector<double> samples)
{
    std::sort(samples.begin(), samples.end());
    return samples.at(samples.size() / 2);
}

static bool same_matrix(const Matrix &a, const Matrix &b)
{
    if (a.rows() != b.rows() || a.cols() != b.cols())
        return false;

    for (std::size_t i = 0; i < a.rows(); ++i)
        for (std::size_t j = 0; j < a.cols(); ++j)
            if (a(i, j) != b(i, j))
                return false;

    return true;
}

static void print_samples(std::string_view name,
                          const std::vector<double> &samples)
{
    std::cout << name << " мс:";
    for (double sample : samples)
        std::cout << ' ' << sample;
    std::cout << '\n';
}

int main(int argc, char **argv)
{
    try
    {
        std::cout << std::fixed << std::setprecision(6);
        const std::string_view build_type = LAB4_BUILD_TYPE;

        const bool benchmark = argc == 2 && std::string_view(argv[1]) == "--benchmark";
        if (argc != 1 && !benchmark)
        {
            throw std::invalid_argument("Использование: lab4 [--benchmark]");
        }
        if (benchmark && build_type != "Release")
        {
            throw std::runtime_error("Измерения требуют отдельной Release-сборки без санитайзеров");
        }

        struct PreparedData
        {
            Matrix x_train, x_test;
            std::vector<double> y_train, y_test;
            std::vector<std::string> feature_names, number_to_label;
        };

        PreparedData data;
        std::vector<double> preparation_times;
        const int repeats = benchmark ? 5 : 1;
        for (int repeat = 0; repeat < repeats; ++repeat)
        {
            PreparedData candidate;
            const auto start = Clock::now();
            {
                Dataset dataset = load_dataset("data/data.csv");
                auto [train_dataset, test_dataset] = dataset.split(0.8);

                std::unordered_set<std::string> included_columns;
                for (const auto &name : dataset.get_feature_names())
                {
                    if (name != "id" && name != "Category")
                    {
                        included_columns.insert(name);
                    }
                }

                std::unordered_map<std::string, double> label_to_number;
                std::vector<std::string> number_to_label;

                std::vector<double> y_train;
                std::vector<double> y_test;

                const auto &train_categories =
                    std::get<CategoricalColumn>(
                        train_dataset.get_columns().at("Category"))
                        .values;

                const auto &test_categories =
                    std::get<CategoricalColumn>(
                        test_dataset.get_columns().at("Category"))
                        .values;

                for (const auto &category : train_categories)
                {
                    if (!category.has_value())
                    {
                        throw std::runtime_error("Пропуск в Category (train)");
                    }

                    const std::string &name = *category;
                    if (!label_to_number.contains(name))
                    {
                        double label = static_cast<double>(number_to_label.size());
                        label_to_number.emplace(name, label);
                        number_to_label.push_back(name);
                    }
                    y_train.push_back(label_to_number.at(name));
                }

                for (const auto &category : test_categories)
                {
                    if (!category.has_value())
                    {
                        throw std::runtime_error("Пропуск в Category (test)");
                    }

                    if (!label_to_number.contains(*category))
                    {
                        throw std::runtime_error("Неизвестный класс в тестовой выборке");
                    }
                    y_test.push_back(label_to_number.at(*category));
                }

                auto train_info = train_dataset.analyze();
                train_dataset.impute(train_info, NumericImputation::Median);
                test_dataset.impute(train_info, NumericImputation::Median);

                auto transformed_train = train_dataset.transform(
                    train_info,
                    Scaling::Robust,
                    Encoding::OneHot,
                    Layout::RowMajor,
                    included_columns);

                auto transformed_test = test_dataset.transform(
                    train_info,
                    Scaling::Robust,
                    Encoding::OneHot,
                    Layout::RowMajor,
                    included_columns);

                if (transformed_train.feature_names != transformed_test.feature_names)
                {
                    throw std::runtime_error("Признаки train и test не совпадают");
                }

                std::vector<std::string> feature_names = std::move(transformed_train.feature_names);
                Matrix X_train = std::move(transformed_train.matrix);
                Matrix X_test = std::move(transformed_test.matrix);

                candidate = PreparedData{std::move(X_train), std::move(X_test),
                                         std::move(y_train), std::move(y_test),
                                         std::move(feature_names), std::move(number_to_label)};
            }
            const auto stop = Clock::now();
            preparation_times.push_back(
                std::chrono::duration<double, std::milli>(stop - start).count());

            if (repeat == 0)
            {
                data = std::move(candidate);
            }
            else if (data.y_train != candidate.y_train || data.y_test != candidate.y_test ||
                     data.feature_names != candidate.feature_names ||
                     data.number_to_label != candidate.number_to_label ||
                     !same_matrix(data.x_train, candidate.x_train) ||
                     !same_matrix(data.x_test, candidate.x_test))
            {
                throw std::runtime_error("Подготовка данных не воспроизводится");
            }
        }

        std::cout << "train=" << data.x_train.rows() << ", test=" << data.x_test.rows()
                  << ", features=" << data.x_train.cols()
                  << ", classes=" << data.number_to_label.size() << ", layout=RowMajor\n";

        RandomForestParams params;
        params.n_estimators = 100;
        params.bootstrap = true;
        params.max_samples = std::nullopt;
        params.max_features = {FeatureMode::Sqrt, 1};
        params.max_depth = 3;
        params.min_samples_split = 2;
        params.min_samples_leaf = 1;
        params.random_state = 42;
        params.n_jobs = 4;

        if (benchmark)
        {
            std::cout << "Измерение производительности: 5 повторов\n";
            print_parameters("Sqrt", 1);

            std::cout << "Изменяется только n_jobs: 1, 2, 4.\n"
                      << "Медианное время подготовки данных, мс: "
                      << median(preparation_times) << '\n';

            print_samples("Подготовка данных", preparation_times);

            struct Measurement
            {
                std::size_t jobs;
                double fit, predict;
            };
            std::vector<Measurement> results;
            std::vector<double> expected_predictions, expected_classes;

            for (std::size_t jobs : {1u, 2u, 4u})
            {
                std::vector<double> fit_times, predict_times;

                for (int repeat = 0; repeat < 5; ++repeat)
                {
                    RandomForestParams benchmark_params = params;
                    benchmark_params.n_jobs = jobs;
                    RandomForestClassifier forest(benchmark_params);

                    fit_times.push_back(measure_ms([&]
                                                   { forest.fit(data.x_train, data.y_train); }));

                    std::vector<double> predictions;
                    predict_times.push_back(measure_ms([&]
                                                       { predictions = forest.predict(data.x_test); }));

                    const auto classes = forest.classes();
                    if (jobs == 1 && repeat == 0)
                    {
                        expected_predictions = predictions;
                        expected_classes.assign(classes.begin(), classes.end());
                    }

                    if (predictions != expected_predictions ||
                        !std::equal(classes.begin(), classes.end(), expected_classes.begin(), expected_classes.end()))
                        throw std::runtime_error("Ответы при разных n_jobs не совпадают");
                }

                results.push_back({jobs, median(fit_times), median(predict_times)});

                std::cout << "\nn_jobs=" << jobs << '\n';
                print_samples("fit", fit_times);
                print_samples("predict", predict_times);
            }

            std::cout << "\n| n_jobs | Медиана обучения, мс | Медиана предсказания, мс | Ускорение обучения | Ускорение предсказания |\n";

            for (const auto &result : results)
                std::cout << "| " << result.jobs << " | " << result.fit << " | " << result.predict
                          << " | " << results.front().fit / result.fit
                          << " | " << results.front().predict / result.predict << " |\n";
            return 0;
        }

        std::cout << "Время подготовки данных (загрузка CSV, разделение выборки, кодирование меток, статистики обучающей выборки, заполнение пропусков, преобразование признаков), мс: "
                  << preparation_times.front() << '\n';

        {
            print_parameters("Sqrt", 4);
            CartClassifier cart(3, 2, 1);
            RandomForestClassifier forest(params);

            const double cart_fit = measure_ms([&]
                                               { cart.fit(data.x_train, data.y_train); });
            const double forest_fit = measure_ms([&]
                                                 { forest.fit(data.x_train, data.y_train); });

            std::vector<double> cart_predictions, forest_predictions;
            const double cart_predict = measure_ms([&]
                                                   { cart_predictions = cart.predict(data.x_test); });
            const double forest_predict = measure_ms([&]
                                                     { forest_predictions = forest.predict(data.x_test); });

            std::cout << "Accuracy: CART=" << accuracy(cart_predictions, data.y_test)
                      << ", Forest=" << accuracy(forest_predictions, data.y_test) << '\n';

            std::cout << "\nCART статистика: ";
            print_stats(cart.stats());
            std::cout << '\n';

            print_forest_stats(forest.tree_stats());
            std::cout << "\nВремя (мс): CART fit=" << cart_fit << ", predict=" << cart_predict
                      << "; Forest fit=" << forest_fit << ", predict=" << forest_predict << "\n\n";

            const Matrix cart_probabilities = cart.predict_proba(data.x_test);
            const Matrix forest_probabilities = forest.predict_proba(data.x_test);

            for (std::size_t i = 0; i < std::min<std::size_t>(5, data.y_test.size()); ++i)
            {
                const auto label_name = [&](double label) -> const std::string &
                { return data.number_to_label.at(static_cast<std::size_t>(label)); };

                std::cout << "Object " << i << ": true=" << label_name(data.y_test[i])
                          << ", CART=" << label_name(cart_predictions[i])
                          << ", Forest=" << label_name(forest_predictions[i]) << '\n';

                std::cout << "  CART вероятности: ";
                print_probabilities(data.number_to_label, cart.classes(), cart_probabilities, i);

                std::cout << "  Forest вероятности: ";
                print_probabilities(data.number_to_label, forest.classes(), forest_probabilities, i);
            }
        }
        // синтетический пример для регрессии
        {
            const auto start = Clock::now();
            Matrix x_train(8, 1, Layout::RowMajor), x_test(4, 1, Layout::RowMajor);
            const std::vector<double> y_train{10, 10, 30, 30, 10, 10, 30, 30};
            const std::vector<double> y_test{10, 30, 10, 30};

            for (std::size_t i = 0; i < x_train.rows(); ++i)
                x_train(i, 0) = double(i + 1);
            for (std::size_t i = 0; i < x_test.rows(); ++i)
                x_test(i, 0) = 1.5 + 2 * i;

            const double preparation = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
            std::cout << "\nСинтетическая регрессия: train=8, test=4, features=1, layout=RowMajor\n";
            print_parameters("All", 4);

            CartRegressor cart(3, 2, 1);
            RandomForestParams regression_params = params;
            regression_params.max_features = {FeatureMode::All, 1};
            RandomForestRegressor forest(regression_params);

            const double cart_fit = measure_ms([&]
                                               { cart.fit(x_train, y_train); });
            const double forest_fit = measure_ms([&]
                                                 { forest.fit(x_train, y_train); });

            std::vector<double> cart_predictions, forest_predictions;
            const double cart_predict = measure_ms([&]
                                                   { cart_predictions = cart.predict(x_test); });
            const double forest_predict = measure_ms([&]
                                                     { forest_predictions = forest.predict(x_test); });

            double cart_error = 0, forest_error = 0;
            for (std::size_t i = 0; i < y_test.size(); ++i)
            {
                cart_error += std::abs(cart_predictions[i] - y_test[i]);
                forest_error += std::abs(forest_predictions[i] - y_test[i]);
                std::cout << "x=" << x_test(i, 0) << ", true=" << y_test[i]
                          << ", CART=" << cart_predictions[i] << ", Forest=" << forest_predictions[i] << '\n';
            }

            std::cout << "\nMAE: CART=" << cart_error / y_test.size()
                      << ", Forest=" << forest_error / y_test.size() << '\n';

            std::cout << "\nCART статистика: ";
            print_stats(cart.stats());
            std::cout << '\n';

            print_forest_stats(forest.tree_stats());
            std::cout << "\nВремя (мс): preparation=" << preparation << "; CART fit=" << cart_fit
                      << ", predict=" << cart_predict << "; Forest fit=" << forest_fit
                      << ", predict=" << forest_predict << '\n';
        }
    }
    catch (const std::exception &error)
    {
        std::cerr << "Ошибка: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
