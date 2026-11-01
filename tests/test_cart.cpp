#include "cart_classifier.hpp"
#include "cart_regressor.hpp"
#include "decision_stump.hpp"
#include "exceptions.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <concepts>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

bool almost_equal(double a, double b)
{
    return std::isfinite(a) && std::isfinite(b) &&
           std::abs(a - b) <= 1e-10 * std::max({1.0, std::abs(a), std::abs(b)});
}

void assert_vectors_equal(std::span<const double> actual,
                          std::span<const double> expected)
{
    assert(actual.size() == expected.size());

    for (std::size_t i = 0; i < expected.size(); ++i)
        assert(almost_equal(actual[i], expected[i]));
}

Matrix make_matrix(std::initializer_list<std::initializer_list<double>> rows,
                   Layout layout = Layout::RowMajor)
{
    assert(rows.size() > 0);
    Matrix result(rows.size(), rows.begin()->size(), layout);

    std::size_t i = 0;
    for (const auto &row : rows)
    {
        assert(row.size() == result.cols());
        std::size_t j = 0;
        for (double number : row)
        {
            result(i, j) = number;
            ++j;
        }

        ++i;
    }
    return result;
}

using ConstMatrix = const Matrix &;

using MutableSlice = MatrixSlice<double>;
using ConstSlice = MatrixSlice<const double>;

// Константная матрица возвращает срезы только для чтения.
static_assert(std::same_as<decltype(std::declval<ConstMatrix>()(0, std::nullopt)),
                           ConstSlice>);

static_assert(std::same_as<decltype(std::declval<ConstMatrix>()(std::nullopt, 0)),
                           ConstSlice>);

// Изменяемый срез возвращает ссылку на изменяемый элемент.
static_assert(std::same_as<decltype(std::declval<MutableSlice>()(0)),
                           double &>);

// Константный срез возвращает константную ссылку.
static_assert(std::same_as<decltype(std::declval<ConstSlice>()(0)),
                           const double &>);

// Изменять элемент константного среза нельзя.
static_assert(!std::is_assignable_v<decltype(std::declval<ConstSlice>()(0)),
                                    double>);

// Разыменование возвращает span только для чтения.
static_assert(std::same_as<decltype(*std::declval<ConstSlice>()),
                           std::span<const double>>);

// Проверяем возможность обращения к элементу матрицы.
template <class T>
concept CanReadElement = requires(T &&m) { std::forward<T>(m)(0, 0); };

// Проверяем возможность получения строки.
template <class T>
concept CanReadRow = requires(T &&m) { std::forward<T>(m)(0, std::nullopt); };

// Проверяем возможность получения столбца.
template <class T>
concept CanReadColumn = requires(T &&m) { std::forward<T>(m)(std::nullopt, 0); };

static_assert(CanReadElement<Matrix &> && CanReadElement<const Matrix &>);
static_assert(CanReadRow<Matrix &> && CanReadRow<const Matrix &>);
static_assert(CanReadColumn<Matrix &> && CanReadColumn<const Matrix &>);

static_assert(!CanReadElement<Matrix> && !CanReadElement<const Matrix>);
static_assert(!CanReadRow<Matrix> && !CanReadRow<const Matrix>);
static_assert(!CanReadColumn<Matrix> && !CanReadColumn<const Matrix>);

void test_matrix_slices()
{
    for (Layout layout : {Layout::RowMajor, Layout::ColumnMajor})
    {
        Matrix x = make_matrix({{1, 2, 3}, {4, 5, 6}}, layout);
        auto row = x(1, std::nullopt);
        auto col = x(std::nullopt, 1);

        const Matrix &read_only = x;
        auto const_row = read_only(1, std::nullopt);
        auto const_col = read_only(std::nullopt, 1);

        assert(row.axis() == SliceAxis::Row && row.size() == 3);
        assert(col.axis() == SliceAxis::Column && col.size() == 2);
        assert(row(0) == 4 && row.at(2) == 6);
        assert(col(0) == 2 && col.at(1) == 5);
        assert(const_row(1) == 5 && const_col(0) == 2);

        if (layout == Layout::RowMajor)
        {
            assert(row.stride() == 1 && row.is_contiguous());
            assert(col.stride() == 3 && !col.is_contiguous());
            assert(row.as_span()[2] == 6 && (*row)[0] == 4);
        }
        else
        {
            assert(row.stride() == 2 && !row.is_contiguous());
            assert(col.stride() == 1 && col.is_contiguous());
            assert(col.as_span()[1] == 5 && (*col)[0] == 2);
        }

        // исключения при непрерывности
        bool caught = false;
        try
        {
            if (layout == Layout::RowMajor)
                (void)col.as_span();
            else
                (void)row.as_span();
        }
        catch (const NonContiguousSliceError &)
        {
            caught = true;
        }
        assert(caught);

        // изменение через срезы
        row(0) = 40;
        assert(x(1, 0) == 40 && const_row(0) == 40);
        col(0) = 20;
        assert(x(0, 1) == 20 && const_col(0) == 20);
        assert(row.as_const()(0) == 40);

        // выход за границы
        caught = false;
        try
        {
            (void)row.at(3);
        }
        catch (const std::out_of_range &)
        {
            caught = true;
        }
        assert(caught);

        caught = false;
        try
        {
            (void)x(2, std::nullopt);
        }
        catch (const std::out_of_range &)
        {
            caught = true;
        }
        assert(caught);

        caught = false;
        try
        {
            (void)x(std::nullopt, 3);
        }
        catch (const std::out_of_range &)
        {
            caught = true;
        }
        assert(caught);
    }
}

void test_empty_and_single_slices()
{
    for (Layout layout : {Layout::RowMajor, Layout::ColumnMajor})
    {
        Matrix no_columns(2, 0, layout);
        Matrix no_rows(0, 3, layout);

        auto row = no_columns(0, std::nullopt);
        auto col = no_rows(std::nullopt, 1);

        assert(row.empty() && col.empty());
        assert(row.data() == nullptr && col.data() == nullptr);
        assert(row.is_contiguous() && col.is_contiguous());
        assert(row.as_span().empty() && col.as_span().empty());
        assert((*row).empty() && (*col).empty());

        const Matrix &a = no_columns, &b = no_rows;
        assert(a(0, std::nullopt).as_span().empty());
        assert(b(std::nullopt, 1).as_span().empty());

        bool caught = false;
        try
        {
            (void)row.at(0);
        }
        catch (const std::out_of_range &)
        {
            caught = true;
        }
        assert(caught);

        // срез из 1-ого элемента
        Matrix one_row(1, 3, layout);
        one_row(0, 1) = 42;
        assert(one_row(std::nullopt, 1).as_span()[0] == 42);

        Matrix one_col(3, 1, layout);
        one_col(1, 0) = 17;
        assert(one_col(1, std::nullopt).as_span()[0] == 17);
    }

    double number = 8;
    MatrixSlice<double> one(&number, 1, 99, SliceAxis::Column);
    assert(one.is_contiguous() && one.as_span()[0] == 8);

    MatrixSlice<double> empty;
    assert(empty.empty() && empty.data() == nullptr);

    bool caught = false;
    try
    {
        MatrixSlice<double> invalid(nullptr, 1, 1, SliceAxis::Row);
        (void)invalid;
    }
    catch (const std::invalid_argument &)
    {
        caught = true;
    }
    assert(caught);
    caught = false;
    try
    {
        MatrixSlice<double> invalid(&number, 1, 0, SliceAxis::Row);
        (void)invalid;
    }
    catch (const std::invalid_argument &)
    {
        caught = true;
    }
    assert(caught);
}

// независимый способ вычислить оценку одной группы
double group_score(const std::vector<double> &group, SplitCriterion criterion)
{
    if (criterion == SplitCriterion::Variance)
    {
        double sum = std::accumulate(group.begin(), group.end(), 0.0);
        return sum * sum / group.size();
    }

    std::map<double, int> counts;
    for (double label : group)
        ++counts[label];

    double score = 0;
    for (const auto &[label, count] : counts)
    {
        (void)label;
        score += static_cast<double>(count) * count;
    }
    return score / group.size();
}

std::optional<Split> slow_best_split(const Matrix &x,
                                     const std::vector<double> &y,
                                     const std::vector<std::size_t> &rows,
                                     std::vector<std::size_t> features,
                                     SplitCriterion criterion,
                                     std::size_t min_leaf)
{
    if (criterion == SplitCriterion::Gini)
    {
        bool one_class = true;
        for (std::size_t i : rows)
            if (y[i] != y[rows[0]])
                one_class = false;

        if (one_class)
            return std::nullopt;
    }

    std::sort(features.begin(), features.end());
    std::optional<Split> best;

    for (std::size_t feature : features)
    {
        std::vector<double> values;
        for (std::size_t i : rows)
            values.push_back(x(i, feature));

        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end()), values.end());

        for (std::size_t k = 1; k < values.size(); ++k)
        {
            double lo = values[k - 1], hi = values[k];
            double threshold = std::midpoint(lo, hi);
            if (threshold == hi)
                threshold = lo;

            std::vector<double> left, right;
            for (std::size_t i : rows)
            {
                if (x(i, feature) <= threshold)
                    left.push_back(y[i]);
                else
                    right.push_back(y[i]);
            }

            if (left.size() < min_leaf || right.size() < min_leaf)
                continue;

            double score =
                group_score(left, criterion) + group_score(right, criterion);

            if (!best || score > best->score ||
                (score == best->score &&
                 (feature < best->feature ||
                  (feature == best->feature && threshold < best->threshold))))
                best = Split{feature, threshold, score};
        }
    }
    return best;
}

void test_split_finder()
{
    Matrix x = make_matrix({{0, 0}, {1, 0}, {2, 1}, {3, 1}, {4, 0}, {5, 1}});
    std::vector<double> classification{10, 10, 42, 42, 100, 100};
    std::vector<double> regression{10, 10, 30, 30, 10, 10};
    SplitFinder finder;

    for (SplitCriterion criterion : {SplitCriterion::Gini, SplitCriterion::Variance})
    {
        const std::vector<double> &y = criterion == SplitCriterion::Gini ? classification : regression;
        for (std::size_t min_leaf : {1u, 2u, 4u})
        {
            for (const std::vector<std::size_t> &ids :
                 {std::vector<std::size_t>{0, 1, 2, 3, 4, 5}, std::vector<std::size_t>{5, 1, 3, 0}})
            {
                std::vector<std::size_t> features{1, 0};

                auto expected = slow_best_split(x, y, ids, features, criterion, min_leaf);
                auto actual = finder.find_best(x, y, ids, features, criterion, min_leaf);

                assert(actual.has_value() == expected.has_value());
                if (expected)
                {
                    assert(actual->feature == expected->feature);
                    assert(almost_equal(actual->score, expected->score));

                    std::size_t left_count = 0;
                    for (std::size_t id : ids)
                    {
                        bool left_actual = x(id, actual->feature) <= actual->threshold;
                        bool left_expected = x(id, expected->feature) <= expected->threshold;

                        assert(left_actual == left_expected);
                        if (left_actual)
                            ++left_count;
                    }
                    assert(left_count >= min_leaf && ids.size() - left_count >= min_leaf);
                }
            }
        }
    }

    Matrix two_features = make_matrix({{0, 0}, {1, 0}, {2, 1}, {3, 0}});
    std::vector<double> y{10, 10, 42, 42};

    std::vector<std::size_t> ids{0, 1, 2, 3};
    std::vector<std::size_t> only_second{1};
    std::vector<std::size_t> all{1, 0};

    auto restricted = finder.find_best(two_features, y, ids, only_second, SplitCriterion::Gini, 1);
    auto unrestricted = finder.find_best(two_features, y, ids, all, SplitCriterion::Gini, 1);

    assert(restricted && restricted->feature == 1);
    assert(unrestricted && unrestricted->feature == 0);
    assert(unrestricted->score > restricted->score);

    std::vector<std::size_t> no_features;
    assert(!finder.find_best(two_features, y, ids, no_features, SplitCriterion::Gini, 1));
    assert(!finder.find_best(two_features, y, ids, all, SplitCriterion::Gini, 3));

    Matrix ties = make_matrix({{0, 0}, {1, 1}, {2, 2}, {3, 3}});
    std::vector<double> symmetric{0, 1, 1, 0};

    auto tie = finder.find_best(ties, symmetric, ids, all, SplitCriterion::Gini, 1);
    assert(tie && tie->feature == 0 && almost_equal(tie->threshold, 0.5));
}

void test_close_double_thresholds()
{
    double a = 1.0;
    double b = std::nextafter(a, std::numeric_limits<double>::infinity());
    for (const auto &[lo, hi] : {std::pair{a, b},
                                 std::pair{0.0, std::numeric_limits<double>::denorm_min()},
                                 std::pair{std::nextafter(std::numeric_limits<double>::max(), 0.0),
                                           std::numeric_limits<double>::max()}})
    {
        Matrix x = make_matrix({{lo}, {lo}, {hi}, {hi}});
        std::vector<double> y{10, 10, 42, 42};
        std::vector<std::size_t> ids{0, 1, 2, 3}, features{0};

        for (SplitCriterion criterion : {SplitCriterion::Gini, SplitCriterion::Variance})
        {
            auto split = SplitFinder{}.find_best(x, y, ids, features, criterion, 2);
            assert(split);
            assert(split->threshold >= lo && split->threshold < hi);
            assert(x(0, 0) <= split->threshold && x(2, 0) > split->threshold);
        }

        CartRegressor regressor(3, 2, 2);
        CartClassifier classifier(3, 2, 2);

        regressor.fit(x, y);
        classifier.fit(x, y);

        assert_vectors_equal(regressor.predict(x), y);
        assert_vectors_equal(classifier.predict(x), y);
    }
}

void test_depth_zero()
{
    Matrix x = make_matrix({{0}, {1}, {2}, {3}});
    std::vector<double> y{10, 42, 100, 10};

    CartRegressor regressor(0, 2, 1);
    CartClassifier classifier(0, 2, 1);
    regressor.fit(x, y);
    classifier.fit(x, y);

    assert(regressor.stats().node_count == 1 && regressor.stats().leaf_count == 1);
    assert(regressor.stats().depth == 0 && regressor.first_splits(5).empty());
    assert_vectors_equal(regressor.predict(x), std::vector<double>(4, 40.5));

    assert(classifier.stats().node_count == 1 && classifier.stats().leaf_count == 1);
    assert(classifier.first_splits(5).empty());
    assert_vectors_equal(classifier.classes(), std::vector<double>{10, 42, 100});
    assert_vectors_equal(classifier.predict(x), std::vector<double>(4, 10));

    Matrix p = classifier.predict_proba(x);
    assert(p.rows() == 4 && p.cols() == 3);
    for (std::size_t i = 0; i < p.rows(); ++i)
    {
        assert(almost_equal(p(i, 0), 0.5));
        assert(almost_equal(p(i, 1), 0.25));
        assert(almost_equal(p(i, 2), 0.25));
    }
}

void test_depth_one_matches_stump()
{
    Matrix x = make_matrix({{0}, {1}, {2}, {3}, {4}, {5}});
    std::vector<double> y{10, 10, 10, 42, 42, 42};

    for (SplitCriterion criterion : {SplitCriterion::Variance, SplitCriterion::Gini})
    {
        DecisionStump stump(criterion, 2);
        CartTree tree(criterion, 1, 2, 2);

        stump.fit(x, y);
        tree.fit(x, y);

        assert(tree.stats().node_count == 3);
        assert(tree.stats().leaf_count == 2);
        assert(tree.stats().depth == 1);

        auto splits = tree.first_splits(10);
        assert(splits.size() == 1);
        assert(splits[0].feature == stump.split_feature());
        assert(almost_equal(splits[0].threshold, stump.split_threshold()));
        assert(tree.first_splits(0).empty());

        if (criterion == SplitCriterion::Variance)
        {
            CartRegressor regressor(1, 2, 2);
            regressor.fit(x, y);
            assert_vectors_equal(regressor.predict(x), stump.predict(x));
        }
        else
        {
            CartClassifier classifier(1, 2, 2);
            classifier.fit(x, y);
            assert_vectors_equal(classifier.predict(x), stump.predict(x));
        }
    }
}

void test_deep_tree()
{
    Matrix x = make_matrix({{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}});
    std::vector<double> y{10, 10, 30, 30, 10, 10, 30, 30};

    CartRegressor shallow(1, 2, 1);
    CartRegressor deep(3, 2, 1);

    shallow.fit(x, y);
    deep.fit(x, y);

    assert(deep.stats().node_count == 7);
    assert(deep.stats().leaf_count == 4);
    assert(deep.stats().depth == 3);
    assert(deep.stats().node_count > shallow.stats().node_count);
    assert(deep.predict(x) != shallow.predict(x));
    assert_vectors_equal(deep.predict(x), y);
    assert(deep.first_splits(10).size() == 3);
    assert(deep.first_splits(2).size() == 2);

    CartClassifier classifier(3, 2, 1);
    classifier.fit(x, y);
    assert(classifier.stats().depth == 3 && classifier.stats().leaf_count == 4);
    assert_vectors_equal(classifier.predict(x), y);
}

void test_stopping_rules()
{
    Matrix x = make_matrix({{0}, {0}, {1}, {1}});
    std::vector<double> y{10, 10, 42, 42};
    std::vector<double> constant{42, 42, 42, 42};
    for (SplitCriterion criterion : {SplitCriterion::Variance, SplitCriterion::Gini})
    {
        CartTree tree(criterion, 8, 2, 1);
        tree.fit(x, constant);
        assert(tree.stats().node_count == 1);

        CartTree min_split(criterion, 8, 5, 1);
        min_split.fit(x, y);
        assert(min_split.stats().node_count == 1);

        CartTree min_leaf(criterion, 8, 2, 3);
        min_leaf.fit(x, y);
        assert(min_leaf.stats().node_count == 1);

        Matrix same_feature(4, 1, Layout::ColumnMajor);
        tree.fit(same_feature, y);
        assert(tree.stats().node_count == 1);

        Matrix one = make_matrix({{5}});
        tree.fit(one, std::vector<double>{42});
        assert(tree.stats().node_count == 1);
        assert(tree.find_leaf(one, 0).is_leaf);

        tree.fit(x, y);
        assert(tree.stats().node_count == 3);
        assert(tree.find_leaf(x, 0).is_valid());
        assert(&tree.find_leaf(x, 0) == &tree.find_leaf(x, 1));
        assert(&tree.find_leaf(x, 2) == &tree.find_leaf(x, 3));
        assert(&tree.find_leaf(x, 0) != &tree.find_leaf(x, 2));
    }

    CartClassifier classifier(8, 2, 1);
    classifier.fit(x, constant);
    assert_vectors_equal(classifier.classes(), std::vector<double>{42});
    assert_vectors_equal(classifier.predict(x), constant);

    Matrix one_class_probabilities = classifier.predict_proba(x);
    assert(almost_equal(one_class_probabilities(0, 0), 1));

    DecisionStump stump(SplitCriterion::Variance, 1);
    stump.fit(x, constant);
    assert(stump.has_split());
}

void test_probabilities()
{
    Matrix x = make_matrix({{0}, {0}, {0}, {1}, {1}, {1}});
    std::vector<double> y{42, 10, 10, 100, 100, 42};

    CartClassifier classifier(3, 2, 1);
    classifier.fit(x, y);

    assert_vectors_equal(classifier.classes(), std::vector<double>{10, 42, 100});
    assert_vectors_equal(classifier.predict(x), std::vector<double>{10, 10, 10, 100, 100, 100});

    Matrix p = classifier.predict_proba(x);
    assert(p.rows() == 6 && p.cols() == 3);

    auto labels = classifier.classes();
    auto predictions = classifier.predict(x);

    for (std::size_t i = 0; i < 6; ++i)
    {
        // в первом листе распределение (2/3, 1/3, 0), во втором (0, 1/3, 2/3)
        assert(almost_equal(p(i, 0), i < 3 ? 2.0 / 3 : 0.0));
        assert(almost_equal(p(i, 1), 1.0 / 3));
        assert(almost_equal(p(i, 2), i < 3 ? 0.0 : 2.0 / 3));

        double sum = 0;
        std::size_t best = 0;
        for (std::size_t j = 0; j < labels.size(); ++j)
        {
            assert(p(i, j) >= 0 && p(i, j) <= 1);
            sum += p(i, j);

            if (p(i, j) > p(i, best))
                best = j;
        }
        assert(almost_equal(sum, 1.0));
        assert(predictions[i] == labels[best]);
    }

    CartClassifier tie(0, 2, 1);
    tie.fit(x, std::vector<double>{100, 42, 10, 100, 42, 10});
    assert_vectors_equal(tie.predict(x), std::vector<double>(6, 10));
}

void test_regressor_errors_and_refit()
{
    Matrix x = make_matrix({{0}, {1}, {2}, {3}});
    std::vector<double> y{10, 10, 42, 42};
    CartRegressor model(3, 2, 1);

    bool caught = false;
    try
    {
        (void)model.predict(x);
    }
    catch (const std::logic_error &)
    {
        caught = true;
    }
    assert(caught);

    model.fit(x, y);

    Matrix empty(0, 1, Layout::RowMajor);
    assert(model.predict(empty).empty());

    // повторное обучение
    Matrix replacement = make_matrix({{1, 3}, {2, 4}});
    model.fit(replacement, std::vector<double>{100, 100});

    assert(model.stats().node_count == 1);
    assert_vectors_equal(model.predict(replacement), std::vector<double>{100, 100});

    caught = false;
    try
    {
        (void)model.predict(x);
    }
    catch (const std::invalid_argument &)
    {
        caught = true;
    }
    assert(caught);

    for (int case_number = 0; case_number < 5; ++case_number)
    {
        Matrix bad_x = x;
        std::vector<double> bad_y = y;
        if (case_number == 0)
            bad_x = Matrix(0, 1, Layout::RowMajor);
        if (case_number == 1)
            bad_x = Matrix(4, 0, Layout::RowMajor);
        if (case_number == 2)
            bad_y = {10};
        if (case_number == 3)
            bad_x(0, 0) = std::numeric_limits<double>::quiet_NaN();
        if (case_number == 4)
            bad_y[0] = std::numeric_limits<double>::infinity();
        model.fit(x, y);

        caught = false;
        try
        {
            model.fit(bad_x, bad_y);
        }
        catch (const std::invalid_argument &)
        {
            caught = true;
        }
        assert(caught);

        // ошибка -> необученнная
        caught = false;
        try
        {
            (void)model.predict(x);
        }
        catch (const std::logic_error &)
        {
            caught = true;
        }
        assert(caught);
    }
}

void test_classifier_errors_and_refit()
{
    Matrix x = make_matrix({{0}, {1}, {2}, {3}});
    std::vector<double> y{10, 10, 42, 42};

    CartClassifier model(3, 2, 1);

    bool caught = false;
    try
    {
        (void)model.classes();
    }
    catch (const std::logic_error &)
    {
        caught = true;
    }
    assert(caught);

    caught = false;
    try
    {
        (void)model.predict_proba(x);
    }
    catch (const std::logic_error &)
    {
        caught = true;
    }
    assert(caught);

    model.fit(x, y);
    Matrix empty(0, 1, Layout::RowMajor);
    assert(model.predict(empty).empty());

    Matrix p = model.predict_proba(empty);
    assert(p.rows() == 0 && p.cols() == 2);

    Matrix replacement = make_matrix({{1, 3}, {2, 4}});
    model.fit(replacement, std::vector<double>{100, 100});

    assert_vectors_equal(model.classes(), std::vector<double>{100});
    assert_vectors_equal(model.predict(replacement), std::vector<double>{100, 100});

    caught = false;
    try
    {
        (void)model.predict_proba(x);
    }
    catch (const std::invalid_argument &)
    {
        caught = true;
    }
    assert(caught);

    model.fit(x, y);
    caught = false;
    try
    {
        model.fit(x, std::vector<double>{10});
    }
    catch (const std::invalid_argument &)
    {
        caught = true;
    }
    assert(caught);

    caught = false;
    try
    {
        (void)model.classes();
    }
    catch (const std::logic_error &)
    {
        caught = true;
    }
    assert(caught);
}

void test_invalid_parameters_and_predictions()
{
    bool caught = false;
    try
    {
        CartRegressor bad(3, 1, 1);
        (void)bad;
    }
    catch (const std::invalid_argument &)
    {
        caught = true;
    }
    assert(caught);

    caught = false;
    try
    {
        CartClassifier bad(3, 2, 0);
        (void)bad;
    }
    catch (const std::invalid_argument &)
    {
        caught = true;
    }
    assert(caught);

    Matrix x = make_matrix({{0}, {1}, {2}, {3}});
    std::vector<double> y{10, 10, 42, 42};

    Matrix invalid = x;
    invalid(1, 0) = std::numeric_limits<double>::infinity();

    CartRegressor regressor(3, 2, 1);
    CartClassifier classifier(3, 2, 1);
    regressor.fit(x, y);
    classifier.fit(x, y);

    caught = false;
    try
    {
        (void)regressor.predict(invalid);
    }
    catch (const std::invalid_argument &)
    {
        caught = true;
    }
    assert(caught);

    caught = false;
    try
    {
        (void)classifier.predict_proba(invalid);
    }
    catch (const std::invalid_argument &)
    {
        caught = true;
    }
    assert(caught);
}

void test_layout_equivalence()
{
    Matrix row = make_matrix({{0, 7, 1}, {1, 7, 0}, {2, 5, 1}, {3, 5, 0}, {4, 3, 1}, {5, 3, 0}, {6, 1, 1}, {7, 1, 0}});
    Matrix col(row.rows(), row.cols(), Layout::ColumnMajor);
    for (std::size_t i = 0; i < row.rows(); ++i)
        for (std::size_t j = 0; j < row.cols(); ++j)
            col(i, j) = row(i, j);
    std::vector<double> y{10, 10, 42, 42, 100, 100, 10, 10};

    CartRegressor ra(3, 2, 1), rb(3, 2, 1);
    ra.fit(row, y);
    rb.fit(col, y);
    assert_vectors_equal(ra.predict(row), rb.predict(col));
    assert(ra.stats().node_count == rb.stats().node_count);
    assert(ra.stats().depth == rb.stats().depth);

    CartClassifier ca(3, 2, 1), cb(3, 2, 1);
    ca.fit(row, y);
    cb.fit(col, y);
    assert_vectors_equal(ca.predict(row), cb.predict(col));
    assert_vectors_equal(ca.classes(), cb.classes());

    auto sa = ca.first_splits(10), sb = cb.first_splits(10);
    assert(sa.size() == sb.size());
    for (std::size_t i = 0; i < sa.size(); ++i)
    {
        assert(sa[i].feature == sb[i].feature);
        assert(sa[i].depth == sb[i].depth);
        assert(almost_equal(sa[i].threshold, sb[i].threshold));
    }

    Matrix pa = ca.predict_proba(row), pb = cb.predict_proba(col);
    for (std::size_t i = 0; i < pa.rows(); ++i)
        for (std::size_t j = 0; j < pa.cols(); ++j)
            assert(almost_equal(pa(i, j), pb(i, j)));
}

int main()
{
    // Проверка срезов строк и столбцов при RowMajor и ColumnMajor:
    // направление, размер, шаг, константность, изменение элементов,
    // получение span и исключения при неплотных срезах и неверных индексах.
    test_matrix_slices();

    // Проверка пустых и одноэлементных срезов:
    // их корректного состояния, непрерывности, доступа к данным
    // и исключений при неправильных параметрах.
    test_empty_and_single_slices();

    // Проверка поиска лучшего разбиения сравнением с медленным перебором:
    // оба критерия, ограничения на листья, подмножества объектов и признаков,
    // правильное распределение объектов и выбор при равных оценках.
    test_split_finder();

    // Проверка порогов между соседними представимыми числами double:
    // порог должен правильно разделять объекты даже при ошибках округления.
    test_close_double_thresholds();

    // Проверка дерева с max_depth = 0:
    // один лист без разбиений, среднее для регрессии,
    // самый частый класс и вероятности для классификации.
    test_depth_zero();

    // Проверка совпадения CART глубины 1 с DecisionStump:
    // одинаковые признаки, пороги, предсказания и структура дерева.
    test_depth_one_matches_stump();

    // Проверка роста дерева на несколько уровней:
    // количество узлов и листьев, глубина, предсказания
    // и отличие глубокого дерева от дерева глубины 1.
    test_deep_tree();

    // Проверка условий остановки дерева:
    // одинаковые ответы и признаки, один объект, ограничения
    // min_samples_split и min_samples_leaf, правильный поиск листьев.
    test_stopping_rules();

    // Проверка вероятностей классификатора:
    // порядок классов, частоты классов в листьях, сумму вероятностей,
    // согласованность predict() с predict_proba() и выбор при ничьей.
    test_probabilities();

    // Проверка ошибок и повторного обучения регрессора:
    // predict() до fit(), пустая выборка, неверные размеры,
    // некорректные данные и сброс состояния после ошибки fit().
    test_regressor_errors_and_refit();

    // Проверка ошибок и повторного обучения классификатора:
    // classes() и predict_proba() до fit(), пустая выборка,
    // обновление классов, неверные размеры и сброс состояния.
    test_classifier_errors_and_refit();

    // Проверка некорректных параметров конструктора
    // и запрета предсказаний при бесконечных значениях признаков.
    test_invalid_parameters_and_predictions();

    // Проверка одинаковой работы CART при RowMajor и ColumnMajor:
    // совпадение разбиений, статистики деревьев, предсказаний
    // и вероятностей классов с учётом погрешности double.
    test_layout_equivalence();
    std::cout << "Все 13 тестов CART прошли успешно\n";
}
