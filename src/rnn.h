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

	RNN(size_t input_dim, size_t output_dim, size_t hidden_dim) {
	    whh = gen_random_matrix(hidden_dim, hidden_dim);
	    wxh = gen_random_matrix(hidden_dim, input_dim);
	    why = gen_random_matrix(output_dim, hidden_dim);

	    bh = gen_zeros_matrix(hidden_dim, 1);
	    by = gen_zeros_matrix(output_dim, 1);
	}
	
	// x is a matrix where each col is a feature and each
	// row is a 
	//
	// y is dim output_dim x 1
	std::tuple<matrix, matrix> forward(matrix &x) {
	    size_t whh_n_rows = std::get<0>(whh).size() / std::get<1>(whh);
	    matrix h = gen_zeros_matrix(whh_n_rows, 1);
	    
	    std::vector<element_type> x_vals = std::get<0>(x);
	    dim x_dim = std::get<1>(x);
	    
	    //size_t x_n_rows = x_vals.size() / x_dim;
	    
	    /// for backwards phase
	    prior_inputs = x;
	    
	    prior_hs = transpose(h);
	    //append_rows(prior_hs, h);

	    for (size_t col = 0; col < x_dim; col++) {
		// avoid memory allocation here/
		matrix x_col = get_col(x, col);

		matrix sum = dot(wxh, x_col);
		matrix t_1 = dot(whh, h);
		add_in_place(sum, t_1);
		add_in_place(sum, h);

		tanh_e_wise(sum);
		// h then is 
		h = sum;

		/// for backwards phase
		// possibly need to figure this out, make more efficient
		//matrix h_T = transpose(h);
		append_cols(prior_hs, h);
	    }
	    
	    matrix y = dot(why, h);
	    add_in_place(y, by);
	    
	    // TODO: can return h vals as well
	    return std::make_tuple(y, h);
	}

	// dy matrix will have dim m x 1
	void backward(matrix &dy) {
	    double learning_rate = 0.02;
	    size_t n_rows = std::get<0>(prior_inputs).size() / std::get<1>(prior_inputs);

	    matrix dwhy = dot(dy, get_row(prior_hs, n_rows));
	    matrix dby = dy;

	    // get shapes
	    dim dwhh_dim = std::get<1>(whh);
	    matrix dwhh = gen_zeros_matrix(std::get<0>(whh).size() / dwhh_dim, dwhh_dim);

	    dim dwxh_dim = std::get<1>(wxh);
	    matrix dwxh = gen_zeros_matrix(std::get<0>(wxh).size() / dwxh_dim, dwxh_dim);

	    dim dbh_dim = std::get<1>(bh);
	    matrix dbh = gen_zeros_matrix(std::get<0>(bh).size() / dbh_dim, dbh_dim);

	    matrix dh = dot(transpose(why), dy);

	    // backpropagate	
	    for (size_t idx = n_rows; idx > -1; idx--) {
		// dbh += (1 - get_row(prior_hs, n_rows) ** 2) * dh
		matrix h_row_temp = get_row(prior_hs, n_rows);
		pow_e_wise(h_row_temp, 2L);
		// this mult then add could be condensed to 1 op, scalar - matrix
		multiply_scalar(h_row_temp, -1);
		add_scalar(h_row_temp, 1);
		multiply(h_row_temp, dh);
		
		add_in_place(dbh, h_row_temp);

		matrix h_row = get_row(prior_hs, n_rows);

		add_in_place(dwhh, dot(h_row_temp, transpose(h_row)));
		
		add_in_place(dwxh, dot(h_row_temp, transpose(h_row)));

		dh = dot(whh, h_row_temp);
	    }
	    // clip
	    clip(dwxh, -1, 1);
	    clip(dwhh, -1, 1);
	    clip(dwhy, -1, 1);
	    clip(dbh, -1, 1);
	    clip(dby, -1, 1);
	    
	    // update weights and biases
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

	double loss(matrix &x, matrix &y) {
	    std::tuple<matrix, matrix> result = forward(x);
	    matrix out = std::get<0>(result);
	    
	    softmax_in_place(out);

	    std::vector<element_type> out_vals = std::get<0>(out);

	    double loss = 0.0;

	    for (element_type val : out_vals) {
		
	    }
	}

	double total_loss() {

	}
};

