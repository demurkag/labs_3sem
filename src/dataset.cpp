#include "dataset.hpp"

#include <string>
#include <vector>
#include <optional>
#include <variant>
#include <unordered_map>
#include <filesystem>
#include <fstream>
#include <charconv>
#include <format>
#include <cmath>
#include <algorithm>
#include <ranges>
#include <stdexcept>
#include <cstddef>
#include <utility>
#include <numeric>
#include <random>

double quantile(std::span<const double> sorted_values, double p)
{
    if (p < 0.0 || p > 1.0)
    {
        throw std::invalid_argument("Квантиль p должен находиться в диапазоне [0, 1]");
    }

    if (sorted_values.empty())
    {
        return 0.0;
    }

    double pos = (sorted_values.size() - 1) * p;

    if (pos == std::floor(pos))
    {
        return sorted_values[static_cast<std::size_t>(pos)];
    }

    std::size_t lo = static_cast<std::size_t>(std::floor(pos));
    double f = pos - lo;

    return sorted_values[lo] + f * (sorted_values[lo + 1] - sorted_values[lo]);
}

double scale_value(
    double x,
    const NumericFeatureInfo &info,
    Scaling scaling)
{
    switch (scaling)
    {
    case Scaling::MinMax:
    {
        double denom = info.max - info.min;
        if (denom == 0.0)
            return 0.0;
        return (x - info.min) / denom;
    }

    case Scaling::Standard:
    {
        double std_dev = std::sqrt(info.variance);
        if (std_dev == 0.0)
            return 0.0;
        return (x - info.mean) / std_dev;
    }

    case Scaling::Robust:
    {
        double denom = info.q75 - info.q25;
        if (denom == 0.0)
            return 0.0;
        return (x - info.median) / denom;
    }

    case Scaling::None:
    {
        return x;
    }
    }

    return 0.0;
}

void encode_categorial(
    const std::vector<std::optional<std::string>> &values,
    const std::vector<std::string> &sorted_categories,
    Encoding encoding,
    const std::string &name,
    std::vector<std::vector<double>> &result_columns,
    std::vector<std::string> &feature_names)
{
    std::size_t num_rows = values.size();

    switch (encoding)
    {
    case Encoding::OneHot:
    {
        for (const std::string &category : sorted_categories)
        {
            std::vector<double> one_hot_column(num_rows, 0.0); // один столбец

            for (std::size_t i = 0; i < num_rows; ++i)
            {
                if (values[i].has_value() && *values[i] == category)
                {
                    one_hot_column[i] = 1.0;
                }
            }

            result_columns.push_back(std::move(one_hot_column));
            feature_names.push_back(name + '_' + category);
        }

        break;
    }

    case Encoding::Ordinal:
    {
        std::vector<double> encoded_values(num_rows);

        for (std::size_t i = 0; i < num_rows; ++i)
        {
            if (values[i].has_value())
            {
                auto it = std::find(sorted_categories.begin(), sorted_categories.end(), *values[i]);
                if (it != sorted_categories.end())
                {
                    encoded_values[i] = static_cast<double>(std::distance(sorted_categories.begin(), it));
                }
            }
        }

        result_columns.push_back(std::move(encoded_values));
        feature_names.push_back(name);
        break;
    }
    }
}

void Dataset::add_column(const std::string &name, const Column &col)
{
    if (name.empty())
    {
        throw std::runtime_error("Нельзя добавить столбец с пустым именем");
        // todo: раскрутка стека
    }

    if (columns_.contains(name))
    {
        throw std::runtime_error("Дублирующееся имя столбца: " + name);
    }

    std::size_t new_size = 0;

    if (std::holds_alternative<NumericColumn>(col))
    {
        new_size = std::get<NumericColumn>(col).values.size();
    }
    else
    {
        new_size =
            std::get<CategoricalColumn>(col).values.size();
    }

    if (!columns_.empty() && new_size != get_num_rows())
    {
        throw std::runtime_error("Все столбцы должны содержать одинаковое число строк");
    }

    columns_.emplace(name, col);
    feature_order_.push_back(name);
}

std::size_t Dataset::get_num_rows() const
{
    if (columns_.empty())
    {
        return 0;
    }

    const auto &first_col = columns_.begin()->second;

    if (std::holds_alternative<NumericColumn>(first_col))
    {
        return std::get<NumericColumn>(first_col).values.size();
    }
    else
    {
        return std::get<CategoricalColumn>(first_col).values.size();
    }
}

// строка -> вектор
std::vector<std::string> split(
    const std::string &line,
    char delimiter)
{
    std::vector<std::string> tokens;
    std::string current;

    for (char c : line)
    {
        if (delimiter == c)
        {
            tokens.push_back(current);
            current.clear();
        }
        else
        {
            current += c;
        }
    }
    tokens.push_back(current);

    return tokens;
}

Dataset load_dataset(
    const std::filesystem::path &filepath,
    char delimiter,
    const std::string &missing_marker)
{
    std::ifstream file(filepath);

    if (!file.is_open())
    {
        throw std::runtime_error("Не удалось открыть файл: " + filepath.string());
    }

    std::string line;

    if (!std::getline(file, line))
    {
        throw std::runtime_error("Файл пуст или не содержит заголовка");
    }

    if (!line.empty() && line.back() == '\r')
    {
        line.pop_back();
    }

    auto headers = split(line, delimiter);
    std::size_t num_cols = headers.size();

    if (num_cols == 0)
    {
        throw std::runtime_error("Заголовок не содержит столбцов");
    }

    std::vector<std::vector<std::string>> raw_columns(num_cols); // временное хранилище

    std::size_t current_row = 1;

    while (std::getline(file, line))
    {
        if (line.empty())
        {
            continue;
        }

        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }

        auto tokens = split(line, delimiter);

        if (tokens.size() != num_cols)
        {
            throw std::runtime_error("Ошибка в строке " + std::to_string(current_row + 1) +
                                     ": ожидалось " + std::to_string(num_cols) +
                                     " полей, найдено " + std::to_string(tokens.size()));
        }

        for (std::size_t j = 0; j < num_cols; ++j)
        {
            raw_columns[j].push_back(tokens[j]); // 2-ое значение в конец 2-ой строки
        }

        current_row++;
    }

    Dataset dataset;

    for (std::size_t j = 0; j < num_cols; ++j)
    {
        std::string col_name = strip_quotes(headers[j]);

        if (col_name.empty())
        {
            col_name = "id";
        }

        auto &raw_data = raw_columns[j]; // один столбец

        for (std::string &value : raw_data)
        {
            value = strip_quotes(value);
        }

        bool is_numeric = true;

        for (const std::string &val : raw_data)
        {
            if (val == missing_marker || val.empty())
            {
                continue;
            }

            double test = 0.0; // std::from_chars сюда запишет преобразование
            auto [ptr, ec] = std::from_chars(val.data(), val.data() + val.size(), test);

            if (ec != std::errc{} || ptr != val.data() + val.size()) // нулевой код ошибки и указатель на последний символ
            {
                is_numeric = false;
                break;
            }
        }

        if (is_numeric)
        {
            NumericColumn num_col;

            for (const std::string &val : raw_data)
            {
                if (val.empty() || val == missing_marker)
                {
                    num_col.values.push_back(std::nullopt);
                }
                else
                {
                    double parsed = 0.0;
                    auto [ptr, ec] = std::from_chars(val.data(), val.data() + val.size(), parsed);

                    if (ec != std::errc{} ||
                        ptr != val.data() + val.size())
                    {
                        throw std::runtime_error("Ошибка разбора числового значения '" +
                                                 val + "' в столбце '" + col_name + "'");
                    }

                    num_col.values.push_back(parsed);
                }
            }
            dataset.add_column(col_name, num_col);
        }
        else
        {
            CategoricalColumn cat_col;
            for (const std::string &val : raw_data)
            {
                if (val.empty() || val == missing_marker)
                {
                    cat_col.values.push_back(std::nullopt);
                }
                else
                {
                    cat_col.values.push_back(val);
                }
            }
            dataset.add_column(col_name, cat_col);
        }
    }

    return dataset;
}

void Dataset::print_summary(std::ostream &out) const
{
    std::size_t num_rows = get_num_rows();

    if (num_rows == 0)
    {
        out << "Датасет пуст: нет объектов для отображения.\n";
        return;
    }

    out << std::format("Объектов: {}\n", num_rows);
    out << std::format("Признаков: {}\n\n", feature_order_.size());

    out << std::format("{:>4} {:<28}{:<20}{:>18}\n", "#", "Признак", "Тип", "Пропуски");

    std::size_t total_missing = 0;
    std::size_t cols_with_missing = 0;

    for (std::size_t index = 0; index < feature_order_.size(); ++index)
    {
        const std::string &name = feature_order_[index];
        const Column &col = columns_.at(name);

        std::string type_str;
        std::size_t missing_count = 0;

        if (std::holds_alternative<NumericColumn>(col))
        {
            type_str = "числовой";
            const auto &num_col = std::get<NumericColumn>(col); // получаем значение по типу

            for (const auto &val : num_col.values)
            {
                if (!val.has_value())
                {
                    missing_count++;
                }
            }
        }
        else
        {
            type_str = "категориальный";
            const auto &cat_col = std::get<CategoricalColumn>(col);

            for (const auto &val : cat_col.values)
            {
                if (!val.has_value())
                {
                    missing_count++;
                }
            }
        }

        std::string missing_str;
        if (missing_count > 0)
        {
            double result = 100.0 * static_cast<double>(missing_count) /
                            static_cast<double>(num_rows);
            missing_str = std::format("{:>10} ({:.1f}%)", missing_count, result);
            total_missing += missing_count;
            cols_with_missing++;
        }
        else
        {
            missing_str = std::format("{:>10}", "0");
        }

        out << std::format("{:>4} {:<20}{:<20}{}\n", index, name, type_str, missing_str);
    }

    out << std::format("\nВсего пропусков: {} в {} признаках\n", total_missing, cols_with_missing);
}

DatasetInfo Dataset::analyze() const
{
    DatasetInfo info;

    for (const auto &[name, col] : columns_)
    {
        if (std::holds_alternative<NumericColumn>(col))
        {
            const auto &num_col = std::get<NumericColumn>(col);
            NumericFeatureInfo num_info{};

            std::vector<double> values; // "чистый" столбец

            for (const auto &val : num_col.values)
            {
                if (val.has_value())
                {
                    values.push_back(*val); // разыменование std::optional
                }
                else
                {
                    num_info.missing_count++;
                }
            }

            if (values.empty())
            {
                info[name] = num_info;
                continue;
            }

            auto [min_it, max_it] = std::ranges::minmax_element(values);
            num_info.min = *min_it;
            num_info.max = *max_it;

            double sum = 0.0;
            for (const auto &val : values)
            {
                sum += val;
            }
            num_info.mean = sum / values.size();

            double var_sum = 0.0;
            for (const auto &val : values)
            {
                double diff = num_info.mean - val;
                var_sum += diff * diff;
            }
            num_info.variance = var_sum / values.size();

            std::ranges::sort(values);

            num_info.q05 = quantile(values, 0.05);
            num_info.q25 = quantile(values, 0.25);
            num_info.median = quantile(values, 0.5);
            num_info.q75 = quantile(values, 0.75);
            num_info.q95 = quantile(values, 0.95);

            info[name] = num_info;
        }
        else
        {
            const auto &cat_col = std::get<CategoricalColumn>(col);
            CategoricalFeatureInfo cat_info{};

            for (const auto &val : cat_col.values)
            {
                if (val.has_value())
                {
                    const std::string &s = *val;
                    cat_info.frequencies[s]++;

                    if (cat_info.frequencies[s] == 1)
                    {
                        cat_info.categories.push_back(s);
                    }
                }
                else
                {
                    cat_info.missing_count++;
                }
            }

            info[name] = cat_info;
        }
    }
    return info;
}

void Dataset::impute(const DatasetInfo &info, NumericImputation strategy)
{
    for (auto &[name, col] : columns_)
    {
        const FeatureInfo &fi = info.at(name);

        if (std::holds_alternative<NumericColumn>(col))
        {
            auto &num_col = std::get<NumericColumn>(col);
            const auto &num_info = std::get<NumericFeatureInfo>(fi);

            double fill_value = 0.0;
            if (strategy == NumericImputation::Mean)
            {
                fill_value = num_info.mean;
            }
            else if (strategy == NumericImputation::Median)
            {
                fill_value = num_info.median;
            }

            for (auto &val : num_col.values)
            {
                if (!val.has_value())
                {
                    val = fill_value;
                }
            }
        }
        else
        {
            auto &cat_col = std::get<CategoricalColumn>(col);

            for (auto &val : cat_col.values)
            {
                if (!val.has_value())
                {
                    val = "__MISSING__";
                }
            }
        }
    }
}

TransformedDataset Dataset::transform(
    const DatasetInfo &info,
    Scaling scaling,
    Encoding encoding,
    Layout layout) const
{
    std::unordered_set<std::string> all_columns(
        feature_order_.begin(),
        feature_order_.end());

    return transform(info, scaling, encoding, layout, all_columns);
}

TransformedDataset Dataset::transform(
    const DatasetInfo &info,
    Scaling scaling,
    Encoding encoding,
    Layout layout,
    const std::unordered_set<std::string> &included_columns) const
{
    if (included_columns.empty())
    {
        throw std::invalid_argument("Не выбраны столбцы для преобразования");
    }

    for (const std::string &name : included_columns)
    {
        if (!columns_.contains(name))
        {
            throw std::invalid_argument("Столбец не найден: " + name);
        }
    }

    std::size_t num_rows = get_num_rows();
    std::vector<std::string> feature_names; // новые названия столбцов при one hot
    std::vector<std::vector<double>> result_columns;

    for (const std::string &name : feature_order_)
    {
        if (!included_columns.contains(name))
        {
            continue;
        }

        const Column &col = columns_.at(name);
        const FeatureInfo &fi = info.at(name);

        if (std::holds_alternative<NumericColumn>(col))
        {
            const auto &num_col = std::get<NumericColumn>(col);
            const auto &num_info = std::get<NumericFeatureInfo>(fi);

            std::vector<double> scaled_values(num_rows); // преобразованный

            for (std::size_t i = 0; i < num_rows; ++i)
            {
                if (num_col.values[i].has_value())
                {
                    scaled_values[i] = scale_value(*num_col.values[i], num_info, scaling);
                }
            }

            result_columns.push_back(std::move(scaled_values));
            feature_names.push_back(name);
        }
        else
        {
            const auto &cat_col = std::get<CategoricalColumn>(col);
            const auto &cat_info = std::get<CategoricalFeatureInfo>(fi);

            std::vector<std::string> sorted_categories = cat_info.categories;

            if (cat_info.missing_count > 0)
            {
                sorted_categories.push_back("__MISSING__");
            }

            std::sort(sorted_categories.begin(), sorted_categories.end());

            encode_categorial(cat_col.values, sorted_categories, encoding, name, result_columns, feature_names);
        }
    }

    std::size_t num_cols = result_columns.size();
    Matrix matrix(num_rows, num_cols, layout);

    for (std::size_t j = 0; j < num_cols; ++j)
        for (std::size_t i = 0; i < num_rows; ++i)
            matrix(i, j) = result_columns[j][i];

    return TransformedDataset{std::move(matrix), std::move(feature_names)};
}

template <typename T>
std::pair<T, T> split_column(
    const T &source_column,
    std::size_t train_rows,
    const std::vector<std::size_t> &indices)
{
    T train_column;
    T test_column;

    for (std::size_t i = 0; i < train_rows; ++i)
    {
        train_column.values.push_back(source_column.values[indices[i]]);
    }

    for (std::size_t i = train_rows; i < indices.size(); ++i)
    {
        test_column.values.push_back(source_column.values[indices[i]]);
    }

    return {std::move(train_column), std::move(test_column)};
}

std::pair<Dataset, Dataset> Dataset::split(double train_ratio) const
{
    if (!(train_ratio > 0.0 && train_ratio < 1.0))
    {
        throw std::invalid_argument(
            "Доля обучающей выборки должна быть больше 0 и меньше 1");
    }

    if (feature_order_.empty())
    {
        throw std::runtime_error(
            "Невозможно разделить датасет без столбцов");
    }

    const Column &first_column = columns_.at(feature_order_.front());

    std::size_t total_rows = 0;

    if (std::holds_alternative<NumericColumn>(first_column))
    {
        total_rows = std::get<NumericColumn>(first_column).values.size();
    }
    else
    {
        total_rows = std::get<CategoricalColumn>(first_column).values.size();
    }

    const std::size_t train_rows = static_cast<std::size_t>(total_rows * train_ratio);

    if (train_rows == 0 || train_rows == total_rows)
    {
        throw std::invalid_argument(
            "При заданной пропорции одна из выборок получается пустой");
    }

    std::vector<std::size_t> indices(total_rows);

    std::iota(indices.begin(), indices.end(), 0);

    std::mt19937 generator(50);
    std::shuffle(indices.begin(), indices.end(), generator);

    Dataset train_dataset;
    Dataset test_dataset;

    train_dataset.feature_order_ = feature_order_;
    test_dataset.feature_order_ = feature_order_;

    for (const auto &name : feature_order_)
    {
        const Column &column = columns_.at(name);

        if (std::holds_alternative<NumericColumn>(column))
        {
            const NumericColumn &source = std::get<NumericColumn>(column);

            if (source.values.size() != total_rows)
            {
                throw std::runtime_error(
                    "Столбцы датасета имеют разное количество строк");
            }

            auto [train_column, test_column] = split_column(source, train_rows, indices);

            train_dataset.columns_.emplace(name, std::move(train_column));

            test_dataset.columns_.emplace(name, std::move(test_column));
        }
        else
        {
            const CategoricalColumn &source = std::get<CategoricalColumn>(column);

            if (source.values.size() != total_rows)
            {
                throw std::runtime_error(
                    "Столбцы датасета имеют разное количество строк");
            }

            auto [train_column, test_column] = split_column(source, train_rows, indices);

            train_dataset.columns_.emplace(name, std::move(train_column));

            test_dataset.columns_.emplace(name, std::move(test_column));
        }
    }

    return {std::move(train_dataset), std::move(test_dataset)};
}

void print_numeric_histogram(
    std::span<const double> values,
    double min,
    double max,
    int num_bins)
{
    if (num_bins <= 0)
    {
        std::cout
            << "Число интервалов должно быть положительным.\n";
        return;
    }

    if (values.empty() || min == max)
    {
        std::cout << "Невозможно построить гистограмму.";
        return;
    }

    double bin_width = (max - min) / num_bins;

    std::vector<int> counts(num_bins, 0);

    for (double v : values)
    {
        int bin = static_cast<int>((v - min) / bin_width); // -> номер интервала

        if (bin >= num_bins)
        {
            bin = num_bins - 1;
        }

        counts[bin]++;
    }

    int max_count = *std::ranges::max_element(counts); // для масштаба

    for (int i = 0; i < num_bins; ++i)
    {
        double bin_start = min + i * bin_width;
        double bin_end = bin_start + bin_width;

        int bar_length = static_cast<int>(30.0 * counts[i] / max_count);

        std::string bar(bar_length, '#');
        std::cout << std::format("  [{:<8.2f}, {:<8.2f})  {:<30}  {}\n",
                                 bin_start, bin_end, bar, counts[i]);
    }
}

void print_categorical_histogram(const CategoricalFeatureInfo &cat_info)
{
    if (cat_info.frequencies.empty())
    {
        std::cout << "Невозможно построить гистограмму.";
        return;
    }

    std::size_t max_freq = 0;
    for (const auto &[cat, freq] : cat_info.frequencies)
    {
        if (freq > max_freq)
        {
            max_freq = freq;
        }
    }

    for (const auto &cat : cat_info.categories)
    {
        int freq = cat_info.frequencies.at(cat); // поиск значения по ключу

        int bar_length = static_cast<int>(30.0 * freq / max_freq);

        std::string bar(bar_length, '#');
        std::cout << std::format("  {:<25}  {:<30}  {}\n", cat, bar, freq);
    }
}

void print_numeric_info(
    std::size_t index,
    const std::string &name,
    const NumericFeatureInfo &info,
    std::span<const double> clean_values,
    int num_bins)
{
    std::cout << std::format("\nПризнак {}: {} (числовой)\n", index, name);
    std::cout << std::format("  пропусков: {}\n", info.missing_count);

    if (clean_values.empty())
    {
        std::cout << "  (Все значения пропущены, статистика недоступна)\n";
        return;
    }

    std::cout << std::format("  min: {:.3f}    max: {:.3f}\n", info.min, info.max);
    std::cout << std::format("  mean: {:.3f}   var: {:.3f}\n", info.mean, info.variance);
    std::cout << std::format("  q05: {:.3f}\n", info.q05);
    std::cout << std::format("  q25: {:.3f}\n", info.q25);
    std::cout << std::format("  q50: {:.3f}\n", info.median);
    std::cout << std::format("  q75: {:.3f}\n", info.q75);
    std::cout << std::format("  q95: {:.3f}\n", info.q95);

    std::cout << "\n  Гистограмма:\n";
    print_numeric_histogram(clean_values, info.min, info.max, num_bins);
}

void print_categorical_info(
    std::size_t index,
    const std::string &name,
    const CategoricalFeatureInfo &info)
{
    std::cout << std::format("\nПризнак {}: {} (категориальный)\n", index, name);
    std::cout << std::format("  пропусков: {}\n", info.missing_count);

    if (info.categories.empty())
    {
        std::cout << "  (Нет категорий)\n";
        return;
    }

    print_categorical_histogram(info);
}