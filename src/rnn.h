#include <math.h>
#include <stdlib.h>
#include <tuple>
#include <vector>

#include "eval.h"
#include "utils.h"

typedef double element_type;
typedef std::tuple<std::vector<element_type>, size_t> matrix;

class RNN {
  public:
    size_t input_dim;
    size_t output_dim;
    size_t hidden_dim;

    matrix whh;
    matrix wxh;
    matrix why;

    matrix bh;
    matrix by;

    // matrix prior_inputs;
    // matrix prior_hs;

    // TODO fix this, probably should be a param input somewhere
    size_t bptt_stop;

    RNN(size_t in_dim, size_t out_dim, size_t hid_dim, size_t bptt_stop_val) {
	bptt_stop = bptt_stop_val;

	input_dim = in_dim;
	output_dim = out_dim;
	hidden_dim = hid_dim;

	whh = gen_random_matrix(hidden_dim, hidden_dim, hidden_dim);
	wxh = gen_random_matrix(hidden_dim, input_dim, input_dim);
	why = gen_random_matrix(output_dim, hidden_dim, input_dim);

	bh = gen_zeros_matrix(hidden_dim, 1);
	by = gen_zeros_matrix(output_dim, 1);
    }

    // x is a matrix where each col is a feature and each
    // row is a
    //
    // y is dim output_dim x 1
    std::tuple<matrix, matrix, matrix, matrix> forward(const matrix &x) {
	size_t whh_n_rows = std::get<0>(whh).size() / std::get<1>(whh);
	matrix h = gen_zeros_matrix(whh_n_rows, 1);

	std::vector<element_type> x_vals = std::get<0>(x);
	dim x_dim = std::get<1>(x);

	// size_t x_n_rows = x_vals.size() / x_dim;

	/// for backwards phase
	matrix prior_inputs = x;

	matrix prior_hs = h;
	// prior_hs = transpose(h);
	// append_rows(prior_hs, h);

	// for (size_t col = 0; col < x_dim; col++) {
	std::vector<element_type> y_vals;

	for (size_t row = 0; row < x_vals.size() / x_dim; row++) {
	    // avoid memory allocation here/
	    matrix x_row = get_row(x, row);
	    //x_row = transpose(x_row);

	    matrix sum = dott(wxh, x_row);

	    matrix t_1 = dot(whh, h);
	    add_in_place(sum, t_1);
	    add_in_place(sum, h);

	    tanh_e_wise(sum);
	    // h then is 64 x 1
	    h = sum;

	    /// for backwards phase
	    // possibly need to figure this out, make more efficient
	    prior_hs = append_cols(prior_hs, h);

	    matrix y = dot(why, h);
	    add_in_place(y, by);
	    // TODO: this is bad
	    y_vals.push_back(std::get<0>(y).at(0));
	}

	// for each time step/hidden state make a prediction

	// note noteNOTE NOTE Oold
	// matrix y = dot(why, h);
	// add_in_place(y, by);

	// TODO: consider removing h here if not needed
	return std::make_tuple(std::make_tuple(y_vals, 1), h, prior_hs, prior_inputs);
    }

    // dy has dim n_samples x output_dim
    void backward(matrix &dy, double learning_rate, matrix &prior_hs, matrix &prior_inputs) {
	// double learning_rate = 0.005;
	size_t n_rows = std::get<0>(prior_inputs).size() / std::get<1>(prior_inputs);

	// matrix dwhy = dot(dy, get_row(prior_hs, n_rows));
	// matrix dby = dy;

	matrix dby;

	dim dwhy_dim = std::get<1>(why);
	matrix dwhy = gen_zeros_matrix(std::get<0>(why).size() / dwhy_dim, dwhy_dim);
	// get shapes
	dim dwhh_dim = std::get<1>(whh);
	matrix dwhh = gen_zeros_matrix(std::get<0>(whh).size() / dwhh_dim, dwhh_dim);

	dim dwxh_dim = std::get<1>(wxh);
	matrix dwxh = gen_zeros_matrix(std::get<0>(wxh).size() / dwxh_dim, dwxh_dim);

	dim dbh_dim = std::get<1>(bh);
	matrix dbh = gen_zeros_matrix(std::get<0>(bh).size() / dbh_dim, dbh_dim);

	// matrix dh = dot(transpose(why), get_row(dy, n_rows));

	// backpropagate
	for (int t_step = n_rows - 1; t_step >= 0; t_step--) {
	    //
	    matrix dh = dot(transpose(why), get_row(dy, t_step));

	    dby = get_row(dy, t_step);
	    matrix last_h = get_col(prior_hs, t_step);
	    add_in_place(dwhy, dott(dby, last_h));

	    // dbh += (1 - get_row(prior_hs, n_rows) ** 2) * dh
	    matrix h_row_temp = get_col(prior_hs, t_step);
	    pow_e_wise(h_row_temp, 2L);
	    // this mult then add could be condensed to 1 op, scalar - matrix
	    multiply_scalar(h_row_temp, -1);
	    add_scalar(h_row_temp, 1);
	    multiply(h_row_temp, dh);
	    add_in_place(dbh, h_row_temp);
	    matrix h_row = get_col(prior_hs, t_step);
	    add_in_place(dwhh, dott(h_row_temp, h_row));
	    matrix prior_input = get_row(prior_inputs, t_step);
	    // TODO: new, resolved error, but bad, fix this
	    prior_input = transpose(prior_input);

	    add_in_place(dwxh, dott(h_row_temp, prior_input));
	    dh = dot(whh, h_row_temp);

	    // TODO: do this better
	    int hard_stop = t_step - bptt_stop;
	    if (t_step - 1 - bptt_stop < -1) {
		hard_stop = -1;
	    }

	    for (int idx = t_step - 1; idx > hard_stop; idx--) {
		matrix dby_temp = get_row(dy, idx);
		matrix last_h = get_col(prior_hs, idx);
		// matrix dwhy_temp = dot(dby, transpose(last_h));
		add_in_place(dwhy, dott(dby, last_h));

		// dbh += (1 - get_row(prior_hs, n_rows) ** 2) * dh
		matrix h_row_temp = get_col(prior_hs, idx);
		pow_e_wise(h_row_temp, 2L);
		// this mult then add could be condensed to 1 op, scalar - matrix
		multiply_scalar(h_row_temp, -1);
		add_scalar(h_row_temp, 1);
		multiply(h_row_temp, dh);
		add_in_place(dbh, h_row_temp);
		matrix h_row = get_col(prior_hs, idx);
		add_in_place(dwhh, dott(h_row_temp, h_row));
		matrix prior_input = get_row(prior_inputs, idx);

		// newnew
		prior_input = transpose(prior_input);

		add_in_place(dwxh, dott(h_row_temp, prior_input));
		dh = dot(whh, h_row_temp);

		// add_in_place(dwhy, dwhy_temp);
		add_in_place(dby, dby_temp);
	    }
	}
	// avg gradient updates

	double n = n_rows;
	multiply_scalar(dwhh, (1.0 / n));
	multiply_scalar(dwxh, (1.0 / n));
	multiply_scalar(dwhy, (1.0 / n));
	multiply_scalar(dbh, (1.0 / n));
	multiply_scalar(dby, (1.0 / n));

	// clip
	clip(dwxh, -1, 1);
	clip(dwhh, -1, 1);
	clip(dwhy, -1, 1);
	clip(dbh, -1, 1);
	clip(dby, -1, 1);

	// update weights and biases, using avg of gradient updates
	multiply_scalar(dwhh, learning_rate);
#pragma omp critical(whh)
	{ subtract_in_place(whh, dwhh); }

	multiply_scalar(dwxh, learning_rate);
#pragma omp critical(wxh)
	{ subtract_in_place(wxh, dwxh); }

	multiply_scalar(dwhy, learning_rate);
#pragma omp critical(why)
	{ subtract_in_place(why, dwhy); }

	multiply_scalar(dbh, learning_rate);
#pragma omp critical(bh)
	{ subtract_in_place(bh, dbh); }

	multiply_scalar(dby, learning_rate);
#pragma omp critical(by)
	{ subtract_in_place(by, dby); }
    }

    // binary cross-entropy
    // y dim = (output_dim x 1)
    // forward output dim = (output_dim x 1)
    double loss(const matrix &x, const matrix &y) {
	std::tuple<matrix, matrix, matrix, matrix> result = forward(x);
	// predictions
	matrix out = std::get<0>(result);

	// not sure about this softmax
	// softmax_in_place(out);
	sigmoid(out);

	clip(out, 0.0001, 0.99999999);

	std::vector<element_type> y_ps = std::get<0>(out);
	std::vector<element_type> y_vals = std::get<0>(y);

	double loss = 0.0;

	for (size_t idx = 0; idx < y_vals.size(); idx++) {
	    const double y_val = y_vals.at(idx);
	    double y_p = y_ps.at(idx);

	    double l = y_val * log(y_p) + (1 - y_val) * log(1 - y_p);

	    loss += l;
	}

	return -1 * loss / y_vals.size();
    }

    double total_loss(const std::vector<matrix> &X, const std::vector<matrix> &Y) {
	double l = 0.0;

	for (size_t idx = 0; idx < Y.size(); idx++) {
	    l += loss(X.at(idx), Y.at(idx));
	}

	return l / Y.size();
    }

    void train(std::vector<matrix> &X, const std::vector<matrix> &Y, size_t num_epochs,
	       double learning_rate) {
	// size_t num_epochs = 30;
	// double learning_rate = 0.0001;

	std::vector<double> losses;

	for (size_t epoch = 0; epoch < num_epochs; epoch++) {
	    shuffle(X);

	    if (epoch > 1 && epoch % 5 == 0) {
		double this_loss = total_loss(X, Y);
		std::cout << "[Epoch " << epoch << "] Total loss: " << this_loss << "\n";

		if (losses.size() > 1 && this_loss > losses.back()) {
		    learning_rate = learning_rate * 0.5;
		}

		losses.push_back(this_loss);
	    }
// for each training example
#pragma omp parallel for
	    for (size_t idx = 0; idx < Y.size(); idx++) {
		std::tuple<matrix, matrix, matrix, matrix> forward_res = forward(X.at(idx));

		// tuple is y, h
		matrix dldy = std::get<0>(forward_res);
		sigmoid(dldy);

		matrix y = Y.at(idx);

		subtract_in_place(dldy, Y.at(idx));

		backward(dldy, learning_rate, std::get<2>(forward_res), std::get<3>(forward_res));
	    }
	}

	save_weights();
    }

    void test(const std::vector<matrix> &X, const std::vector<matrix> &Y) {
	size_t true_pos = 0;
	size_t true_neg = 0;
	size_t false_pos = 0;
	size_t false_neg = 0;

	double threshold = 0.5;

	// for each training example
	for (size_t idx = 0; idx < Y.size(); idx++) {
	    std::tuple<matrix, matrix, matrix, matrix> forward_res = forward(X.at(idx));

	    // tuple is y, h
	    matrix y_hat = std::get<0>(forward_res);
	    sigmoid(y_hat);

	    round(y_hat, threshold);
	    matrix y = Y.at(idx);

	    outcomes counts = count_outcomes(y, y_hat);

	    true_pos += std::get<0>(counts);
	    true_neg += std::get<1>(counts);
	    false_pos += std::get<2>(counts);
	    false_neg += std::get<3>(counts);
	}

	// TODO maybe this tuple is unnecessary
	metrics test_metrics =
	    get_metrics(std::make_tuple(true_pos, true_neg, false_pos, false_neg));

	std::cout << "Accuracy: " << std::get<0>(test_metrics) << "\n";
	std::cout << "Precision: " << std::get<1>(test_metrics) << "\n";
	std::cout << "Recall: " << std::get<2>(test_metrics) << "\n";
	std::cout << "F1: " << std::get<3>(test_metrics) << "\n";
    }

    void save_weights() {
	fs::create_directory("weights");
	write(whh, "weights/whh");
	write(wxh, "weights/wxh");
	write(why, "weights/why");
	write(bh, "weights/bh");
	write(by, "weights/by");
    }

    void load_weights(std::string dir_path) {
	whh = load_weights_matrix("whh", hidden_dim, hidden_dim);
	wxh = load_weights_matrix("wxh", hidden_dim, input_dim);
	why = load_weights_matrix("why", output_dim, hidden_dim);
	bh = load_weights_matrix("bh", hidden_dim, 1);
	by = load_weights_matrix("by", output_dim, 1);
    }
};
