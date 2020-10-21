#include <math.h>
#include <vector>
#include <stdlib.h>
#include <tuple>

#include "utils.h"

typedef double element_type;
typedef std::tuple<std::vector<element_type>, size_t> matrix;

class RNN {
    public:
	matrix whh;
	matrix wxh;
	matrix why;

	matrix bh;
	matrix by;
	
	matrix prior_inputs;
	// shape
	matrix prior_hs;

	int bptt_stop = 4;

	RNN(size_t input_dim, size_t output_dim, size_t hidden_dim) {
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
	std::tuple<matrix, matrix> forward(const matrix &x) {
	    size_t whh_n_rows = std::get<0>(whh).size() / std::get<1>(whh);
	    matrix h = gen_zeros_matrix(whh_n_rows, 1);
	    
	    std::vector<element_type> x_vals = std::get<0>(x);
	    dim x_dim = std::get<1>(x);
	    
	    //size_t x_n_rows = x_vals.size() / x_dim;
	    
	    /// for backwards phase
	    prior_inputs = x;
	    
	    prior_hs = h;
	    //prior_hs = transpose(h);
	    //append_rows(prior_hs, h);

	    //for (size_t col = 0; col < x_dim; col++) {
	    std::vector<element_type> y_vals;

	    for (size_t row = 0; row < x_vals.size() / x_dim; row++) {
            // avoid memory allocation here/
            //matrix x_col = get_col(x, col);
            matrix x_row = get_row(x, row);

            //matrix sum = dot(wxh, x_col);
            matrix sum = dot(wxh, x_row);

            matrix t_1 = dot(whh, h);
            add_in_place(sum, t_1);
            add_in_place(sum, h);

            tanh_e_wise(sum);
            // h then is 64 x 1
            h = sum;

            /// for backwards phase
            // possibly need to figure this out, make more efficient
            //matrix h_T = transpose(h);
            prior_hs = append_cols(prior_hs, h);
            //append_rows(prior_hs, h);

            // that new hotness starts right here
            matrix y = dot(why, h);
            add_in_place(y, by);
            // TODO: this is bad
            y_vals.push_back(std::get<0>(y).at(0));

	    }
	    
	    // for each time step/hidden state make a prediction 
	    
	    // note noteNOTE NOTE Oold
	    //matrix y = dot(why, h);
	    //add_in_place(y, by);
	    
	    // TODO: can return h vals as well
	    //
	    //return std::make_tuple(y, h);
	    return std::make_tuple(std::make_tuple(y_vals, 1), h);
	}

	// dy has dim n_samples x output_dim
	void backward(matrix &dy) {
	    double learning_rate = 0.0005;
	    size_t n_rows = std::get<0>(prior_inputs).size() / std::get<1>(prior_inputs);

	    //matrix dwhy = dot(dy, get_row(prior_hs, n_rows));
	    //matrix dby = dy;
	    
	    matrix dby;
	    matrix dwhy = gen_zeros_matrix(1, 64);
	    // get shapes
	    dim dwhh_dim = std::get<1>(whh);
	    matrix dwhh = gen_zeros_matrix(std::get<0>(whh).size() / dwhh_dim, dwhh_dim);

	    dim dwxh_dim = std::get<1>(wxh);
	    matrix dwxh = gen_zeros_matrix(std::get<0>(wxh).size() / dwxh_dim, dwxh_dim);

	    dim dbh_dim = std::get<1>(bh);
	    matrix dbh = gen_zeros_matrix(std::get<0>(bh).size() / dbh_dim, dbh_dim);

	    //matrix dh = dot(transpose(why), get_row(dy, n_rows));

	    // backpropagate	
	    for (int t_step = n_rows; t_step >= 0; t_step--) {
	        //
            matrix dh = dot(transpose(why), get_row(dy, t_step));

            dby = get_row(dy, t_step);
            matrix last_h = get_col(prior_hs, t_step);
            //dwhy = dot(dby, transpose(last_h));
            add_in_place(dwhy, dot(dby, transpose(last_h)));

            // dbh += (1 - get_row(prior_hs, n_rows) ** 2) * dh
            matrix h_row_temp = get_col(prior_hs, t_step);
            pow_e_wise(h_row_temp, 2L);
            // this mult then add could be condensed to 1 op, scalar - matrix
            multiply_scalar(h_row_temp, -1);
            add_scalar(h_row_temp, 1);
            multiply(h_row_temp, dh);
            add_in_place(dbh, h_row_temp);
            matrix h_row = get_col(prior_hs, t_step);
            add_in_place(dwhh, dot(h_row_temp, transpose(h_row)));
            matrix prior_input = get_row(prior_inputs, t_step);
            add_in_place(dwxh, dot(h_row_temp, transpose(prior_input)));
            dh = dot(whh, h_row_temp);

            // TODO: do this better
            int hard_stop = t_step - bptt_stop;
            if (t_step - 1 - bptt_stop < -1) {
                hard_stop = -1;
            }

            for (int idx = t_step - 1; idx > hard_stop; idx--) {
                matrix dby_temp = get_row(dy, idx);
                matrix last_h = get_col(prior_hs, idx);
                //matrix dwhy_temp = dot(dby, transpose(last_h));
                add_in_place(dwhy, dot(dby, transpose(last_h)));

                // dbh += (1 - get_row(prior_hs, n_rows) ** 2) * dh
                matrix h_row_temp = get_col(prior_hs, idx);
                pow_e_wise(h_row_temp, 2L);
                // this mult then add could be condensed to 1 op, scalar - matrix
                multiply_scalar(h_row_temp, -1);
                add_scalar(h_row_temp, 1);
                multiply(h_row_temp, dh);
                add_in_place(dbh, h_row_temp);
                matrix h_row = get_col(prior_hs, idx);
                add_in_place(dwhh, dot(h_row_temp, transpose(h_row)));
                matrix prior_input = get_row(prior_inputs, idx);
                add_in_place(dwxh, dot(h_row_temp, transpose(prior_input)));
                dh = dot(whh, h_row_temp);

                //add_in_place(dwhy, dwhy_temp);
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
	    //clip(dwxh, -1, 1);
	    //clip(dwhh, -1, 1);
	    //clip(dwhy, -1, 1);
	    //clip(dbh, -1, 1);
	    //clip(dby, -1, 1);
	    
	    // update weights and biases, using avg of gradient updates
	    multiply_scalar(dwhh, learning_rate);
	    subtract_in_place(whh, dwhh);

	    multiply_scalar(dwxh, learning_rate);
	    subtract_in_place(wxh, dwxh);

	    multiply_scalar(dwhy, learning_rate);
	    subtract_in_place(why, dwhy);

	    multiply_scalar(dbh, learning_rate);
	    subtract_in_place(bh, dbh);

	    multiply_scalar(dby, learning_rate);
	    subtract_in_place(by, dby);

	}
	
	// binary cross-entropy
	// y dim = (output_dim x 1)
	// forward output dim = (output_dim x 1)
	double loss(const matrix &x, const matrix &y, const bool flag) {
	    //std::cout<<"forward call\n";
	    std::tuple<matrix, matrix> result = forward(x);
	    // predictions
	    matrix out = std::get<0>(result);
	    
	    // not sure about this softmax
	    //softmax_in_place(out);
	    sigmoid(out);

	    if (flag) {
		    std::cout<<"printing forward probs:\n";
		    print_matrix(out);
	    }

        clip(out, 0.0001, 0.99999999);

	    std::vector<element_type> y_ps = std::get<0>(out);
	    std::vector<element_type> y_vals = std::get<0>(y);

	    double loss = 0.0;

	    for (size_t idx = 0; idx < y_vals.size(); idx++) {
            const double y_val = y_vals.at(idx);
            double y_p = y_ps.at(idx);
            //if (y_p == 0.0 || y_p == 1.0) {
            //    y_p += 0.000001;
            //}

            double l = y_val * log(y_p) + (1 - y_val) * log(1 - y_p);

            loss += l;

	    }

	    return -1 * loss / y_vals.size();
	}
	
	double total_loss(const std::vector<matrix> &X, const std::vector<matrix> &Y,
		const bool flag) {
	    double l = 0.0;

	    for (size_t idx = 0; idx < Y.size(); idx++) {
		    l += loss(X.at(idx), Y.at(idx), flag);
	    }

	    return l / Y.size();
	}

	void train(const std::vector<matrix> &X, const std::vector<matrix> &Y) {
	    size_t num_epochs = 100;
	    
	    //std::vector<element_type> y_vals = std::get<0>(y);

	    for (size_t epoch = 0; epoch < num_epochs; epoch++) {
            if (epoch > 1) {
                std::cout << "Total loss: " << total_loss(X, Y, false) << "\n";
            }
            // for each training example
            for (size_t idx = 0; idx < Y.size(); idx++) {
                std::tuple<matrix, matrix> forward_res = forward(X.at(idx));

                // tuple is y, h
                matrix dldy = std::get<0>(forward_res);
                softmax_in_place(dldy);

                matrix y = Y.at(idx);

                subtract_in_place(dldy, Y.at(idx));

                // for each timestep
                //size_t num_timesteps = std::get<0>(dldy).size();
                //dldy = transpose(dldy);
                //for
                backward(dldy);
            }
	    }
	}

};

