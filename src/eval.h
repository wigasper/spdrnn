#include <tuple>
#include <vector>

// true pos, true neg, false pos, false neg
typedef std::tuple<size_t, size_t, size_t, size_t> outcomes;

typedef double element_type;
typedef std::tuple<std::vector<element_type>, size_t> matrix;
// accuracy, precision, recall, f1
typedef std::tuple<double, double, double, double> metrics;

outcomes count_outcomes(const matrix &y, const matrix &y_hat) {
    std::vector<element_type> y_vals = std::get<0>(y);
    std::vector<element_type> y_hat_vals = std::get<0>(y_hat);

    size_t tp = 0;
    size_t tn = 0;
    size_t fp = 0;
    size_t fn = 0;

    // TODO better way to do this? linear algebra?
    for (size_t idx = 0; idx < y_vals.size(); idx++) {
	if (y_vals.at(idx) == 1 && y_hat_vals.at(idx) == 1) {
	    tp += 1;
	} else if (y_vals.at(idx) == 1 && y_hat_vals.at(idx) == 0) {
	    fn += 1;
	} else if (y_vals.at(idx) == 0 && y_hat_vals.at(idx) == 1) {
	    fp += 1;
	} else {
	    tn += 1;
	}
    }

    return std::make_tuple(tp, tn, fp, fn);
}

metrics get_metrics(const outcomes &counts) {
    double tp = std::get<0>(counts);
    double tn = std::get<1>(counts);
    double fp = std::get<2>(counts);
    double fn = std::get<3>(counts);

    double accuracy = (tp + tn) / (tp + tn + fp + fn);
    double precision = tp / (tp + fp);
    double recall = tp / (tp + fn);
    double f1 = (2 * precision * recall) / (precision + recall);

    return std::make_tuple(accuracy, precision, recall, f1);
}
