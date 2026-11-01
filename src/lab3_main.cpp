#include "dataset.hpp"
#include "cart_classifier.hpp"
#include "cart_regressor.hpp"

#include <algorithm>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

int main()
{
    try
    {
        std::cout << std::fixed << std::setprecision(3); // вывод чисел

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

        std::vector<std::string> feature_names = transformed_train.feature_names;
        Matrix X_train = std::move(transformed_train.matrix);
        Matrix X_test = std::move(transformed_test.matrix);

        std::cout << "CART: классификация\n";
        CartClassifier classifier(3, 2, 1);
        IClassifier &classification_model = classifier;
        classification_model.fit(X_train, y_train);

        TreeStats classification_stats = classifier.stats();
        std::cout << "Обучающих объектов: " << X_train.rows() << '\n';
        std::cout << "Тестовых объектов: " << X_test.rows() << '\n';
        std::cout << "Количество узлов: " << classification_stats.node_count << '\n';
        std::cout << "Количество листьев: " << classification_stats.leaf_count << '\n';
        std::cout << "Фактическая глубина: " << classification_stats.depth << '\n';

        std::cout << "Первые разбиения:\n";
        auto classification_splits = classifier.first_splits(5);
        if (classification_splits.empty())
        {
            std::cout << "  Разбиений нет (один лист)\n";
        }
        for (const SplitInfo &split : classification_splits)
        {
            std::cout << "  Глубина " << split.depth
                      << ": " << feature_names.at(split.feature)
                      << " <= " << split.threshold << '\n';
        }

        std::vector<double> predictions = classification_model.predict(X_test);
        Matrix probabilities = classification_model.predict_proba(X_test);
        auto classes = classification_model.classes();

        std::cout << "\nПервые тестовые предсказания:\n";
        for (std::size_t i = 0;
             i < std::min<std::size_t>(5, predictions.size()); ++i)
        {
            std::size_t predicted_label = static_cast<std::size_t>(predictions[i]);
            std::size_t true_label = static_cast<std::size_t>(y_test[i]);

            std::cout << "Объект " << i
                      << ": предсказано = " << number_to_label.at(predicted_label)
                      << ", правильный ответ = " << number_to_label.at(true_label)
                      << '\n';

            std::cout << "  Вероятности классов: ";
            for (std::size_t c = 0; c < classes.size(); ++c)
            {
                std::size_t label = static_cast<std::size_t>(classes[c]);
                if (c > 0)
                {
                    std::cout << ", ";
                }
                std::cout << number_to_label.at(label)
                          << " = " << probabilities(i, c);
            }
            std::cout << '\n';
        }

        std::cout << "\nCART: синтетическая регрессия\n";

        Matrix X_reg_train(8, 1, Layout::RowMajor);
        std::vector<double> y_reg_train = {10, 10, 30, 30, 10, 10, 30, 30};
        for (std::size_t i = 0; i < 8; ++i)
        {
            X_reg_train(i, 0) = static_cast<double>(i + 1);
        }

        Matrix X_reg_test(4, 1, Layout::RowMajor);
        X_reg_test(0, 0) = 1.5;
        X_reg_test(1, 0) = 3.5;
        X_reg_test(2, 0) = 5.5;
        X_reg_test(3, 0) = 7.5;
        std::vector<double> y_reg_test = {10, 30, 10, 30};

        TreeStats stump_stats;
        TreeStats deep_stats;

        std::vector<double> stump_predictions;
        std::vector<double> deep_predictions;

        for (std::size_t max_depth : {std::size_t{0}, std::size_t{1}, std::size_t{3}})
        {
            CartRegressor regressor(max_depth, 2, 1);
            IRegressor &regression_model = regressor;
            regression_model.fit(X_reg_train, y_reg_train);

            TreeStats stats = regressor.stats();
            std::vector<double> reg_predictions = regression_model.predict(X_reg_test);

            if (max_depth == 1)
            {
                stump_stats = stats;
                stump_predictions = reg_predictions;
            }
            if (max_depth == 3)
            {
                deep_stats = stats;
                deep_predictions = reg_predictions;
            }

            std::cout << "\nmax_depth = " << max_depth << '\n';
            std::cout << "Узлов: " << stats.node_count
                      << ", листьев: " << stats.leaf_count
                      << ", фактическая глубина: " << stats.depth << '\n';

            std::cout << "Первые разбиения:\n";
            auto splits = regressor.first_splits(5);
            if (splits.empty())
            {
                std::cout << "  Разбиений нет (один лист)\n";
            }
            for (const SplitInfo &split : splits)
            {
                std::cout << "  Глубина " << split.depth
                          << ": x <= " << split.threshold << '\n';
            }

            std::cout << "Тестовые предсказания:\n";
            for (std::size_t i = 0; i < reg_predictions.size(); ++i)
            {
                std::cout << "  x = " << X_reg_test(i, 0)
                          << ": предсказано = " << reg_predictions[i]
                          << ", правильный ответ = " << y_reg_test[i] << '\n';
            }
        }

        bool differs = deep_stats.depth > stump_stats.depth &&
                       deep_stats.node_count > stump_stats.node_count &&
                       deep_predictions != stump_predictions;

        std::cout << "\nГлубокое дерево отличается от пня: "
                  << (differs ? "да" : "нет") << '\n';
        if (!differs)
        {
            throw std::runtime_error("Не удалось показать отличие дерева от пня");
        }
    }
    catch (const std::exception &error)
    {
        std::cerr << "Ошибка: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
