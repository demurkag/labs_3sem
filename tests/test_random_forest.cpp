#include "random_forest_classifier.hpp"
#include "random_forest_regressor.hpp"
#include "forest_tasks.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string_view>

namespace
{
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

    void assert_matrices_equal(const Matrix &actual,
                               const Matrix &expected)
    {
        assert(actual.rows() == expected.rows() && actual.cols() == expected.cols());
        for (std::size_t i = 0; i < expected.rows(); ++i)
            for (std::size_t j = 0; j < expected.cols(); ++j)
                assert(almost_equal(actual(i, j), expected(i, j)));
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
            for (double value : row)
                result(i, j++) = value;
            ++i;
        }
        return result;
    }

    template <class Exception, class Function>
    void assert_throws(Function action)
    {
        bool caught = false;
        try
        {
            action();
        }
        catch (const Exception &)
        {
            caught = true;
        }
        assert(caught);
    }

    RandomForestParams small_params()
    {
        RandomForestParams p;
        p.n_estimators = 3;
        p.random_state = 2;
        p.n_jobs = 1;
        p.max_features = {FeatureMode::All, 1};
        return p;
    }

    void test_single_tree_matches_cart()
    {
        Matrix x = make_matrix({{0}, {0}, {2}, {2}});
        const std::vector<double> y{10, 10, 42, 42};
        Matrix query = make_matrix({{-1}, {0}, {2}, {3}});

        auto p = small_params();
        p.n_estimators = 1;
        p.bootstrap = false;
        p.max_depth = 1;
        p.min_samples_leaf = 2;

        RandomForestRegressor r(p);
        RandomForestClassifier c(p);
        CartRegressor cart_r(1, 2, 2);
        CartClassifier cart_c(1, 2, 2);

        r.fit(x, y);
        c.fit(x, y);
        cart_r.fit(x, y);
        cart_c.fit(x, y);

        assert_vectors_equal(r.predict(query), y);
        assert_vectors_equal(c.predict(query), y);
        assert_vectors_equal(r.predict(query), cart_r.predict(query));
        assert_vectors_equal(c.predict(query), cart_c.predict(query));
        assert_vectors_equal(c.classes(), cart_c.classes());
        assert_matrices_equal(c.predict_proba(query), cart_c.predict_proba(query));
    }

    void test_rng_fixture()
    {
        std::mt19937 seeds(2);
        const std::array<std::array<std::size_t, 3>, 3> draws{{{1, 0, 1}, {2, 1, 1}, {2, 0, 2}}};
        for (const auto &expected : draws)
        {
            std::mt19937 rng(seeds());
            std::uniform_int_distribution<std::size_t> distribution(0, 2);
            for (std::size_t index : expected)
                assert(distribution(rng) == index);
        }
    }

    void test_regression_average_and_max_samples()
    {
        Matrix x = make_matrix({{7}, {7}, {7}});
        const std::vector<double> y{0, 3, 12};
        auto p = small_params();
        // план из test_rng_fixture: средние деревьев 2, 6, 8 -> 16/3
        RandomForestRegressor default_samples(p);
        default_samples.fit(x, y);
        assert_vectors_equal(default_samples.predict(x), std::vector<double>(3, 16.0 / 3));
        p.max_samples = 3;
        RandomForestRegressor explicit_samples(p);
        explicit_samples.fit(x, y);
        assert_vectors_equal(explicit_samples.predict(x), default_samples.predict(x));
        // первые два извлечения: средние 1.5, 7.5, 6 -> 5
        p.max_samples = 2;
        RandomForestRegressor two(p);
        two.fit(x, y);
        assert_vectors_equal(two.predict(x), std::vector<double>(3, 5));
        // первые извлечения: ответы 3, 12, 12 -> 9
        p.max_samples = 1;
        RandomForestRegressor one(p);
        one.fit(x, y);
        assert_vectors_equal(one.predict(x), std::vector<double>(3, 9));
    }

    void test_probability_average()
    {
        Matrix x = make_matrix({{7}, {7}, {7}});
        // A=-4.5, B=90. В трёх листьях P(A)=2/3, 2/3, 0
        // Голоса A,A,B дали бы A; среднее P(A)=4/9 даёт B
        const std::vector<double> y{90, -4.5, 90};
        auto p = small_params();
        RandomForestClassifier c(p);
        c.fit(x, y);
        assert_vectors_equal(c.classes(), std::vector<double>{-4.5, 90});
        assert_vectors_equal(c.predict(x), std::vector<double>(3, 90));
        auto probabilities = c.predict_proba(x);
        assert(probabilities.rows() == 3 && probabilities.cols() == 2);
        for (std::size_t i = 0; i < 3; ++i)
        {
            assert(almost_equal(probabilities(i, 0), 4.0 / 9));
            assert(almost_equal(probabilities(i, 1), 5.0 / 9));
        }
        // max_samples=1: каждый bootstrap содержит только один из двух классов.
        p.max_samples = 1;
        RandomForestClassifier missing(p);
        missing.fit(x, y);
        assert_vectors_equal(missing.classes(), std::vector<double>{-4.5, 90});
        auto one = missing.predict_proba(x);
        assert(one.cols() == 2);
        for (std::size_t i = 0; i < 3; ++i)
        {
            assert(almost_equal(one(i, 0), 1.0 / 3));
            assert(almost_equal(one(i, 1), 2.0 / 3));
        }
        assert_vectors_equal(missing.predict(x), std::vector<double>(3, 90));
    }

    void test_exact_probability_average()
    {
        constexpr std::size_t n = 100000;
        constexpr std::size_t m = 100;
        constexpr double class_a = -4.5;
        constexpr double class_b = 90;
        auto p = small_params();
        p.n_estimators = 3;
        p.bootstrap = true;
        p.max_samples = m;
        p.random_state = 0;
        p.n_jobs = 1;

        std::mt19937 seed_rng(p.random_state);
        std::vector<std::uint32_t> seeds;
        while (seeds.size() < p.n_estimators)
        {
            const std::uint32_t seed = seed_rng();
            if (std::find(seeds.begin(), seeds.end(), seed) == seeds.end())
                seeds.push_back(seed);
        }
        std::array<std::array<std::size_t, m>, 3> draws{};
        std::vector<std::size_t> all_indices;
        all_indices.reserve(3 * m);
        for (std::size_t tree = 0; tree < draws.size(); ++tree)
        {
            std::mt19937 rng(seeds[tree]);
            std::uniform_int_distribution<std::size_t> distribution(0, n - 1);
            for (std::size_t &index : draws[tree])
            {
                index = distribution(rng);
                all_indices.push_back(index);
            }
        }
        assert(all_indices.size() == 300);
        std::sort(all_indices.begin(), all_indices.end());
        assert(std::adjacent_find(all_indices.begin(), all_indices.end()) == all_indices.end());

        Matrix x(n, 1, Layout::RowMajor);
        for (std::size_t i = 0; i < n; ++i)
            x(i, 0) = 7;
        std::vector<double> y(n, class_b);
        for (std::size_t tree = 0; tree < 2; ++tree)
            for (std::size_t j = 0; j < 51; ++j)
                y[draws[tree][j]] = class_a;

        // при константном признаке каждое дерево — один лист: P(A)=51/100,51/100,0
        const std::array<std::size_t, 3> expected_a_counts{51, 51, 0};
        for (std::size_t tree = 0; tree < draws.size(); ++tree)
        {
            const auto count = std::count_if(draws[tree].begin(), draws[tree].end(),
                                             [&](std::size_t index)
                                             { return y[index] == class_a; });
            assert(static_cast<std::size_t>(count) == expected_a_counts[tree]);
        }

        RandomForestClassifier model(p);
        model.fit(x, y);
        assert_vectors_equal(model.classes(), std::vector<double>{class_a, class_b});
        Matrix query = make_matrix({{7}, {-1}, {42}});
        const Matrix probabilities = model.predict_proba(query);
        assert(probabilities.rows() == 3 && probabilities.cols() == 2);
        for (std::size_t i = 0; i < query.rows(); ++i)
        {
            assert(almost_equal(probabilities(i, 0), 0.34));
            assert(almost_equal(probabilities(i, 1), 0.66));
        }
        // голосование A,A,B выбрало бы A; усреднение вероятностей выбирает B
        assert_vectors_equal(model.predict(query), std::vector<double>(3, class_b));
    }

    void test_controlled_counts()
    {
        Matrix x = make_matrix({{0}, {1}, {2}});
        const std::vector<double> y{10, 30, 100};
        const std::vector<std::size_t> counts{3, 1, 0};
        for (bool restrict_leaf : {false, true})
        {
            CartRegressor r(std::nullopt, restrict_leaf ? 2 : 3, restrict_leaf ? 2 : 1);
            CartClassifier c(std::nullopt, restrict_leaf ? 2 : 3, restrict_leaf ? 2 : 1);
            r.fit_with_counts(x, y, counts);
            c.fit_with_counts(x, y, counts);
            // Два разных индекса: min_split=3 или min_leaf=2 запрещают разделение, хотя сумма кратностей 4
            assert(r.stats().node_count == 1 && c.stats().node_count == 1);
            assert_vectors_equal(r.predict(x), std::vector<double>(3, 15));
            assert_vectors_equal(c.classes(), std::vector<double>{10, 30, 100});
            auto probabilities = c.predict_proba(x);
            for (std::size_t i = 0; i < 3; ++i)
            {
                assert(almost_equal(probabilities(i, 0), 0.75));
                assert(almost_equal(probabilities(i, 1), 0.25));
                assert(almost_equal(probabilities(i, 2), 0));
            }
        }
        CartRegressor r(std::nullopt, 2, 1);
        CartClassifier c(std::nullopt, 2, 1);
        r.fit_with_counts(x, y, counts);
        c.fit_with_counts(x, y, counts);
        assert_vectors_equal(r.predict(x), std::vector<double>{10, 30, 30});
        assert_vectors_equal(c.predict(x), std::vector<double>{10, 30, 30});
        assert(counts == (std::vector<std::size_t>{3, 1, 0}));
    }

    void test_classes_ties_and_small_inputs()
    {
        auto p = small_params();
        p.bootstrap = false;
        Matrix x = make_matrix({{1}, {1}, {1}, {1}});
        RandomForestClassifier c(p);
        RandomForestRegressor r(p);
        const std::vector<double> y{90, -4.5, 90, -4.5};
        c.fit(x, y);
        r.fit(x, y);
        assert_vectors_equal(c.classes(), std::vector<double>{-4.5, 90});
        assert_vectors_equal(c.predict(x), std::vector<double>(4, -4.5));
        assert_vectors_equal(r.predict(x), std::vector<double>(4, 42.75));
        auto probabilities = c.predict_proba(x);
        for (std::size_t i = 0; i < 4; ++i)
            for (std::size_t j = 0; j < 2; ++j)
                assert(almost_equal(probabilities(i, j), 0.5));
        p.bootstrap = true;
        RandomForestClassifier single_c(p);
        RandomForestRegressor single_r(p);
        Matrix one = make_matrix({{8}});
        const std::vector<double> single_y{-4.5};
        single_c.fit(one, single_y);
        single_r.fit(one, single_y);
        assert_vectors_equal(single_c.classes(), single_y);
        assert_vectors_equal(single_c.predict(one), single_y);
        assert_vectors_equal(single_r.predict(one), single_y);
        probabilities = single_c.predict_proba(one);
        assert(probabilities.rows() == 1 && probabilities.cols() == 1);
        assert(almost_equal(probabilities(0, 0), 1));
        single_c.fit(x, std::vector<double>(4, 90));
        assert_vectors_equal(single_c.classes(), std::vector<double>{90});
        assert_vectors_equal(single_c.predict(x), std::vector<double>(4, 90));
    }

    void test_reproducibility_layout_and_refit()
    {
        Matrix row = make_matrix({{0, 7, 1}, {1, 7, 0}, {2, 5, 1}, {3, 5, 0}, {4, 3, 1}, {5, 3, 0}, {6, 1, 1}, {7, 1, 0}});
        Matrix col(row.rows(), row.cols(), Layout::ColumnMajor);
        for (std::size_t i = 0; i < row.rows(); ++i)
            for (std::size_t j = 0; j < row.cols(); ++j)
                col(i, j) = row(i, j);
        const std::vector<double> y{10, 10, 42, 42, 100, 100, 10, 10};
        for (FeatureSearchParams features : {FeatureSearchParams{FeatureMode::All, 1},
                                             FeatureSearchParams{FeatureMode::Sqrt, 1},
                                             FeatureSearchParams{FeatureMode::Count, 2}})
        {
            auto p = small_params();
            p.n_estimators = 7;
            p.max_features = features;
            RandomForestRegressor ra(p), rb(p);
            RandomForestClassifier ca(p), cb(p);
            ra.fit(row, y);
            rb.fit(col, y);
            ca.fit(row, y);
            cb.fit(col, y);
            const auto rp = ra.predict(row), cp = ca.predict(row);
            const auto probabilities = ca.predict_proba(row);
            assert_vectors_equal(rp, rb.predict(col));
            assert_vectors_equal(rp, ra.predict(col));
            assert_vectors_equal(cp, cb.predict(col));
            assert_vectors_equal(cp, ca.predict(col));
            assert_vectors_equal(ca.classes(), cb.classes());
            assert_matrices_equal(probabilities, cb.predict_proba(col));
            assert_matrices_equal(probabilities, ca.predict_proba(col));
            ra.fit(row, y);
            ca.fit(row, y);
            assert_vectors_equal(rp, ra.predict(row));
            assert_vectors_equal(cp, ca.predict(row));
            assert_matrices_equal(probabilities, ca.predict_proba(row));
            Matrix replacement = make_matrix({{1, 4}, {2, 5}});
            const std::vector<double> new_y{-17, -17};
            ra.fit(replacement, new_y);
            ca.fit(replacement, new_y);
            assert_vectors_equal(ra.predict(replacement), new_y);
            assert_vectors_equal(ca.predict(replacement), new_y);
            assert_vectors_equal(ca.classes(), std::vector<double>{-17});
            assert_throws<std::invalid_argument>([&]
                                                 { (void)ra.predict(row); });
            assert_throws<std::invalid_argument>([&]
                                                 { (void)ca.predict_proba(row); });
        }
    }

    void test_invalid_parameters()
    {
        for (int scenario = 0; scenario < 9; ++scenario)
        {
            auto p = small_params();
            switch (scenario)
            {
            case 0:
                p.n_estimators = 0;
                break;
            case 1:
                p.min_samples_split = 1;
                break;
            case 2:
                p.min_samples_leaf = 0;
                break;
            case 3:
                p.n_jobs = 0;
                break;
            case 4:
                p.max_depth = 0;
                break;
            case 5:
                p.max_samples = 0;
                break;
            case 6:
                p.bootstrap = false;
                p.max_samples = 1;
                break;
            case 7:
                p.max_features = {FeatureMode::Count, 0};
                break;
            case 8:
                p.max_features.mode = static_cast<FeatureMode>(-1);
                break;
            }
            assert_throws<std::invalid_argument>([&]
                                                 { RandomForestRegressor bad(p); });
            assert_throws<std::invalid_argument>([&]
                                                 { RandomForestClassifier bad(p); });
        }
    }

    void assert_untrained(const RandomForestRegressor &r, const RandomForestClassifier &c,
                          const Matrix &x)
    {
        assert_throws<std::logic_error>([&]
                                        { (void)r.predict(x); });
        assert_throws<std::logic_error>([&]
                                        { (void)c.predict(x); });
        assert_throws<std::logic_error>([&]
                                        { (void)c.predict_proba(x); });
        assert_throws<std::logic_error>([&]
                                        { (void)c.classes(); });
    }

    void test_data_errors_and_state()
    {
        Matrix x = make_matrix({{0, 1}, {1, 0}, {2, 1}});
        const std::vector<double> y{10, 30, 100};
        auto p = small_params();
        RandomForestRegressor r(p);
        RandomForestClassifier c(p);
        assert_untrained(r, c, x);
        for (int scenario = 0; scenario < 8; ++scenario)
        {
            Matrix bad_x = x;
            auto bad_y = y;
            switch (scenario)
            {
            case 0:
                bad_x = Matrix(0, 2, Layout::RowMajor);
                bad_y.clear();
                break;
            case 1:
                bad_x = Matrix(3, 0, Layout::RowMajor);
                break;
            case 2:
                bad_y.pop_back();
                break;
            case 3:
                bad_y.push_back(10);
                break;
            case 4:
                bad_x(0, 0) = std::numeric_limits<double>::quiet_NaN();
                break;
            case 5:
                bad_x(0, 0) = -std::numeric_limits<double>::infinity();
                break;
            case 6:
                bad_y[0] = std::numeric_limits<double>::quiet_NaN();
                break;
            case 7:
                bad_y[0] = std::numeric_limits<double>::infinity();
                break;
            }
            r.fit(x, y);
            c.fit(x, y);
            assert_throws<std::invalid_argument>([&]
                                                 { r.fit(bad_x, bad_y); });
            assert_throws<std::invalid_argument>([&]
                                                 { c.fit(bad_x, bad_y); });
            assert_untrained(r, c, x);
        }

        for (bool samples : {false, true})
        {
            p = small_params();
            if (samples)
                p.max_samples = 3;
            else
                p.max_features = {FeatureMode::Count, 2};
            RandomForestRegressor pr(p);
            RandomForestClassifier pc(p);
            pr.fit(x, y);
            pc.fit(x, y);
            Matrix smaller(1, 1, Layout::RowMajor);
            const std::vector<double> single_y{10};
            assert_throws<std::invalid_argument>([&]
                                                 { pr.fit(smaller, single_y); });
            assert_throws<std::invalid_argument>([&]
                                                 { pc.fit(smaller, single_y); });
            assert_untrained(pr, pc, x);
        }
        r.fit(x, y);
        c.fit(x, y); // восстановление после ошибки
        const auto before_r = r.predict(x), before_c = c.predict(x);
        const auto before_p = c.predict_proba(x);
        for (Layout layout : {Layout::RowMajor, Layout::ColumnMajor})
        {
            Matrix empty(0, 2, layout);
            assert(r.predict(empty).empty() && c.predict(empty).empty());
            auto probabilities = c.predict_proba(empty);
            assert(probabilities.rows() == 0 && probabilities.cols() == 3);
        }
        for (int scenario = 0; scenario < 5; ++scenario)
        {
            Matrix bad = x;
            if (scenario == 0)
                bad = Matrix(3, 1, Layout::RowMajor);
            if (scenario == 1)
                bad = Matrix(0, 1, Layout::RowMajor);
            if (scenario == 2)
                bad = Matrix(3, 0, Layout::RowMajor);
            if (scenario == 3)
                bad(0, 0) = std::numeric_limits<double>::quiet_NaN();
            if (scenario == 4)
                bad(0, 0) = std::numeric_limits<double>::infinity();
            assert_throws<std::invalid_argument>([&]
                                                 { (void)r.predict(bad); });
            assert_throws<std::invalid_argument>([&]
                                                 { (void)c.predict(bad); });
            assert_throws<std::invalid_argument>([&]
                                                 { (void)c.predict_proba(bad); });
        }
        assert_vectors_equal(before_r, r.predict(x));
        assert_vectors_equal(before_c, c.predict(x));
        assert_vectors_equal(c.classes(), std::vector<double>{10, 30, 100});
        assert_matrices_equal(before_p, c.predict_proba(x));
    }
    void test_parallel_results()
    {
        const std::vector<double> y{-7.5, -7.5, 42, 90, 42, 90, -7.5, 42};
        for (Layout layout : {Layout::RowMajor, Layout::ColumnMajor})
            for (bool bootstrap : {false, true})
                for (std::size_t tree_count : {1u, 7u})
                {
                    Matrix x = make_matrix({{0, 7, 1}, {1, 7, 0}, {2, 5, 1}, {3, 5, 0}, {4, 3, 1}, {5, 3, 0}, {6, 1, 1}, {7, 1, 0}}, layout);
                    auto p = small_params();
                    p.n_estimators = tree_count;
                    p.bootstrap = bootstrap;
                    p.max_features = {FeatureMode::Sqrt, 1};
                    RandomForestRegressor reference_r(p);
                    RandomForestClassifier reference_c(p);
                    reference_r.fit(x, y);
                    reference_c.fit(x, y);
                    const auto expected_r = reference_r.predict(x);
                    const auto expected_c = reference_c.predict(x);
                    const Matrix expected_p = reference_c.predict_proba(x);
                    const auto expected_classes = reference_c.classes();
                    for (std::size_t jobs : {1u, 2u, 3u, 12u})
                    {
                        p.n_jobs = jobs;
                        RandomForestRegressor r(p);
                        RandomForestClassifier c(p);
                        // Независимые объекты с тем же seed; затем повторный fit
                        // после другого обучения должен восстановить те же ответы.
                        for (int repeat = 0; repeat < 2; ++repeat)
                        {
                            r.fit(x, y);
                            c.fit(x, y);
                            assert(r.predict(x) == expected_r);
                            assert(c.predict(x) == expected_c);
                            assert_vectors_equal(c.classes(), expected_classes);
                            Matrix probabilities = c.predict_proba(x);
                            assert(probabilities.rows() == expected_p.rows());
                            assert(probabilities.cols() == expected_p.cols());
                            for (std::size_t i = 0; i < probabilities.rows(); ++i)
                                for (std::size_t j = 0; j < probabilities.cols(); ++j)
                                    assert(probabilities(i, j) == expected_p(i, j));

                            Matrix empty(0, x.cols(), layout);
                            assert(r.predict(empty).empty());
                            assert(c.predict(empty).empty());
                            probabilities = c.predict_proba(empty);
                            assert(probabilities.rows() == 0);
                            assert(probabilities.cols() == expected_classes.size());
                            assert_vectors_equal(c.classes(), expected_classes);
                            if (repeat == 0)
                            {
                                const std::vector<double> replacement(y.size(), -17);
                                r.fit(x, replacement);
                                c.fit(x, replacement);
                                assert(r.predict(x) == replacement);
                                assert(c.predict(x) == replacement);
                            }
                        }
                    }
                }
    }

    void test_parallel_bootstrap_counts()
    {
        const std::array<std::uint32_t, 3> seeds{1872583848u, 794921487u, 111352301u};
        const std::array<std::vector<std::size_t>, 3> expected_counts{
            std::vector<std::size_t>{1, 2, 0}, {0, 2, 1}, {1, 0, 2}};
        std::mt19937 seed_rng(2);
        for (std::size_t i = 0; i < seeds.size(); ++i)
        {
            assert(seed_rng() == seeds[i]);
            for (std::size_t j = 0; j < i; ++j)
                assert(seeds[i] != seeds[j]);
        }
        const std::vector<double> y{10, 30, 100};
        const std::array<std::vector<double>, 3> split_answers{
            std::vector<double>{10, 30, 30}, {30, 30, 100}, {10, 10, 100}};
        for (Layout layout : {Layout::RowMajor, Layout::ColumnMajor})
            for (bool constant : {false, true})
            {
                Matrix x = constant ? make_matrix({{7}, {7}, {7}}, layout)
                                    : make_matrix({{0}, {1}, {2}}, layout);
                for (std::size_t jobs : {1u, 2u, 3u, 12u})
                    for (bool reverse : {false, true})
                    {
                        std::array<std::vector<std::size_t>, 3> counts;
                        std::array<std::vector<double>, 3> answers;
                        forest_detail::run_tasks(seeds.size(), jobs, [&](std::size_t task)
                                                 {
                            const std::size_t tree = reverse ? seeds.size() - 1 - task : task;
                            std::mt19937 rng(seeds[tree]);
                            std::uniform_int_distribution<std::size_t> draw(0, x.rows() - 1);
                            counts[tree].assign(x.rows(), 0);
                            for (std::size_t k = 0; k < 3; ++k)
                                ++counts[tree][draw(rng)];
                            assert(counts[tree] == expected_counts[tree]);
                            const auto saved = counts[tree];
                            CartRegressor r(std::nullopt, 2, 1, {}, seeds[tree]);
                            CartClassifier c(std::nullopt, 2, 1, {}, seeds[tree]);
                            r.fit_with_counts(x, y, counts[tree]);
                            assert(counts[tree] == saved);
                            c.fit_with_counts(x, y, counts[tree]);
                            assert(counts[tree] == saved);
                            answers[tree] = r.predict(x);
                            const auto rs = r.stats(), cs = c.stats();
                            assert(rs.node_count == (constant ? 1u : 3u));
                            assert(cs.node_count == rs.node_count && cs.leaf_count == rs.leaf_count && cs.depth == rs.depth);
                            if (constant)
                            {
                                const std::array<double, 3> means{70.0 / 3, 160.0 / 3, 70};
                                assert_vectors_equal(answers[tree], std::vector<double>(3, means[tree]));
                                const Matrix probabilities = c.predict_proba(x);
                                assert(probabilities.rows() == 3 && probabilities.cols() == 3);
                                for (std::size_t i = 0; i < 3; ++i)
                                    for (std::size_t label = 0; label < 3; ++label)
                                        assert(almost_equal(probabilities(i, label), saved[label] / 3.0));
                            }
                            else
                            {
                                assert_vectors_equal(answers[tree], split_answers[tree]);
                                assert_vectors_equal(c.predict(x), split_answers[tree]);
                            } });
                        assert(counts == expected_counts);
                        for (const auto &plan : counts)
                            assert(std::accumulate(plan.begin(), plan.end(), std::size_t{0}) == 3);

                        auto p = small_params();
                        p.n_jobs = jobs;
                        RandomForestRegressor r(p);
                        RandomForestClassifier c(p);
                        r.fit(x, y);
                        c.fit(x, y);
                        assert_vectors_equal(r.predict(x), constant ? std::vector<double>(3, 440.0 / 9)
                                                                    : std::vector<double>{50.0 / 3, 70.0 / 3, 230.0 / 3});
                        assert_vectors_equal(c.predict(x), constant ? std::vector<double>(3, 30) : y);
                        assert_vectors_equal(c.classes(), y);
                        const Matrix probabilities = c.predict_proba(x);
                        const Matrix expected = constant ? make_matrix({{2.0 / 9, 4.0 / 9, 1.0 / 3},
                                                                        {2.0 / 9, 4.0 / 9, 1.0 / 3},
                                                                        {2.0 / 9, 4.0 / 9, 1.0 / 3}})
                                                         : make_matrix({{2.0 / 3, 1.0 / 3, 0},
                                                                        {1.0 / 3, 2.0 / 3, 0},
                                                                        {0, 1.0 / 3, 2.0 / 3}});
                        assert_matrices_equal(probabilities, expected);
                        assert(counts == expected_counts);
                    }
            }
    }

    void test_worker_exceptions()
    {
        const auto caller = std::this_thread::get_id();
        for (std::size_t jobs : {1u, 2u, 3u, 12u})
        {
            std::array<int, 7> visits{};
            std::array<std::thread::id, 7> owners{};
            forest_detail::run_tasks(visits.size(), jobs, [&](std::size_t i)
                                     {
                ++visits[i];
                owners[i] = std::this_thread::get_id(); });
            const std::size_t p = std::min(jobs, visits.size());
            for (std::size_t i = 0; i < visits.size(); ++i)
            {
                assert(visits[i] == 1);
                assert(owners[i] == owners[i % p]);
                assert((owners[i] == caller) == (p == 1));
            }
        }

        std::array<int, 7> completed{};
        bool caught = false;
        try
        {
            forest_detail::run_tasks(completed.size(), 3, [&](std::size_t i)
                                     {
                completed[i] = 1;
                if (i == 0 || i == 4)
                    throw std::runtime_error(i == 0 ? "tree zero" : "tree four"); });
        }
        catch (const std::runtime_error &error)
        {
            caught = true;
            assert(std::string_view(error.what()) == "tree zero");
        }
        assert(caught);
        for (int value : completed)
            assert(value == 1);
        assert_throws<int>([]
                           { forest_detail::run_tasks(3, 2, [](std::size_t i)
                                                      {
                if (i == 1)
                    throw 42; }); });

        Matrix x = make_matrix({{0}, {1}, {2}, {3}});
        const std::vector<double> y{10, 10, 42, 42};
        for (std::size_t jobs : {1u, 2u, 3u, 12u})
        {
            auto p = small_params();
            p.n_jobs = jobs;
            p.bootstrap = false;
            RandomForestRegressor r(p);
            RandomForestClassifier c(p);
            r.fit(x, y);
            c.fit(x, y);
            Matrix bad(1, 2, Layout::RowMajor);
            assert_throws<std::invalid_argument>([&]
                                                 { (void)r.predict(bad); });
            assert_throws<std::invalid_argument>([&]
                                                 { (void)c.predict_proba(bad); });
            assert_throws<std::invalid_argument>([&]
                                                 { (void)c.predict(bad); });
            assert(r.predict(x) == y);
            assert(c.predict(x) == y);

            const std::vector<double> huge{-1e200, -1e200, 1e200, 1e200};
            assert_throws<std::runtime_error>([&]
                                              { r.fit(x, huge); });
            assert_throws<std::logic_error>([&]
                                            { (void)r.predict(x); });
            r.fit(x, y);
            assert(r.predict(x) == y);

            assert_throws<std::invalid_argument>([&]
                                                 { c.fit(x, std::vector<double>{10}); });
            assert_throws<std::logic_error>([&]
                                            { (void)c.classes(); });
            assert_throws<std::logic_error>([&]
                                            { (void)c.predict_proba(x); });
            c.fit(x, y);
            assert(c.predict(x) == y);
        }
    }
    void test_tree_stats()
    {
        Matrix x = make_matrix({{0}, {1}, {2}});
        const std::vector<double> y{10, 30, 100};
        // seed=0: bootstrap-план (2,2,0), (1,0,0), (2,2,2)
        // даёт два дерева с одним разбиением, затем одно дерево-лист.
        for (std::size_t jobs : {1u, 3u, 10u})
        {
            auto p = small_params();
            p.random_state = 0;
            p.n_jobs = jobs;
            RandomForestRegressor r(p);
            RandomForestClassifier c(p);
            assert_throws<std::logic_error>([&]
                                            { (void)r.tree_stats(); });
            assert_throws<std::logic_error>([&]
                                            { (void)c.tree_stats(); });
            r.fit(x, y);
            c.fit(x, y);
            for (auto stats : {r.tree_stats(), c.tree_stats()})
            {
                assert(stats.size() == 3);
                for (std::size_t i = 0; i < stats.size(); ++i)
                {
                    assert(stats[i].node_count == (i < 2 ? 3u : 1u));
                    assert(stats[i].leaf_count == (i < 2 ? 2u : 1u));
                    assert(stats[i].depth == (i < 2 ? 1u : 0u));
                }
                stats[0].node_count = 999; // возвращаются копии, без доступа к деревьям
            }
            assert(r.tree_stats()[0].node_count == 3);
            assert(c.tree_stats()[0].node_count == 3);
            const std::vector<double> constant(3, 42);
            r.fit(x, constant);
            c.fit(x, constant);
            for (const auto &stats : {r.tree_stats(), c.tree_stats()})
            {
                assert(stats.size() == p.n_estimators);
                for (const auto &tree : stats)
                    assert(tree.node_count == 1 && tree.leaf_count == 1 && tree.depth == 0);
            }
            assert_throws<std::invalid_argument>([&]
                                                 { r.fit(x, std::vector<double>{10}); });
            assert_throws<std::invalid_argument>([&]
                                                 { c.fit(x, std::vector<double>{10}); });
            assert_throws<std::logic_error>([&]
                                            { (void)r.tree_stats(); });
            assert_throws<std::logic_error>([&]
                                            { (void)c.tree_stats(); });
        }
    }
} // namespace

int main(int argc, char **argv)
{
    const std::array tests{
        std::pair<std::string_view, void (*)()>{"cart", test_single_tree_matches_cart},
        std::pair<std::string_view, void (*)()>{"rng", test_rng_fixture},
        std::pair<std::string_view, void (*)()>{"regression", test_regression_average_and_max_samples},
        std::pair<std::string_view, void (*)()>{"probabilities", test_probability_average},
        std::pair<std::string_view, void (*)()>{"exact_probabilities", test_exact_probability_average},
        std::pair<std::string_view, void (*)()>{"counts", test_controlled_counts},
        std::pair<std::string_view, void (*)()>{"small", test_classes_ties_and_small_inputs},
        std::pair<std::string_view, void (*)()>{"refit", test_reproducibility_layout_and_refit},
        std::pair<std::string_view, void (*)()>{"parameters", test_invalid_parameters},
        std::pair<std::string_view, void (*)()>{"errors", test_data_errors_and_state},
        std::pair<std::string_view, void (*)()>{"parallel", test_parallel_results},
        std::pair<std::string_view, void (*)()>{"parallel_counts", test_parallel_bootstrap_counts},
        std::pair<std::string_view, void (*)()>{"worker_exceptions", test_worker_exceptions},
        std::pair<std::string_view, void (*)()>{"tree_stats", test_tree_stats}};
    bool selected = false;
    for (const auto &[name, test] : tests)
        if (argc == 1 || (argc == 2 && argv[1] == name))
        {
            selected = true;
            test();
            std::cout << "Random forest: " << name << " OK\n";
        }
    return selected ? 0 : 1;
}
