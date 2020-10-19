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
	    //std::cout<<"here0\n";
	    //for (size_t col = 0; col < x_dim; col++) {
	    std::vector<element_type> y_vals;

	    for (size_t row = 0; row < x_vals.size() / x_dim; row++) {
		// avoid memory allocation here/
		//matrix x_col = get_col(x, col);
		matrix x_row = get_row(x, row);	

		//std::cout<<"1\n";
		//matrix sum = dot(wxh, x_col);
		matrix sum = dot(wxh, x_row);
		//std::cout<<"2\n";
		matrix t_1 = dot(whh, h);
		add_in_place(sum, t_1);
		add_in_place(sum, h);

		tanh_e_wise(sum);
		// h then is 64 x 1
		h = sum;

		/// for backwards phase
		// possibly need to figure this out, make more efficient
		//matrix h_T = transpose(h);
		append_cols(prior_hs, h);
		//append_rows(prior_hs, h);
		//std::cout<<"bottomforloopforward\n";

		// that new hotness starts right here
		matrix y = dot(why, h);
		add_in_place(y, by);
		// TODO: this is bad
		y_vals.push_back(std::get<0>(y).at(0));
	    }
	    
	    // for each time step/hidden state make a prediction 
	    
	    // note noteNOTE NOTE Oold
	    //std::cout<<"3\n";
	    //matrix y = dot(why, h);
	    //add_in_place(y, by);
	    
	    // TODO: can return h vals as well
	    //
	    //return std::make_tuple(y, h);
	    return std::make_tuple(std::make_tuple(y_vals, 1), h);
	}

	// dy matrix will have dim output x n_samples
	void backward(matrix &dy) {
	    double learning_rate = 0.02;
	    size_t n_rows = std::get<0>(prior_inputs).size() / std::get<1>(prior_inputs);
	    
	    //std::cout<<"0\n";
	    matrix dwhy = dot(dy, get_row(prior_hs, n_rows));
	    matrix dby = dy;

	    // get shapes
	    dim dwhh_dim = std::get<1>(whh);
	    matrix dwhh = gen_zeros_matrix(std::get<0>(whh).size() / dwhh_dim, dwhh_dim);

	    dim dwxh_dim = std::get<1>(wxh);
	    matrix dwxh = gen_zeros_matrix(std::get<0>(wxh).size() / dwxh_dim, dwxh_dim);

	    dim dbh_dim = std::get<1>(bh);
	    matrix dbh = gen_zeros_matrix(std::get<0>(bh).size() / dbh_dim, dbh_dim);
	    
	    //std::cout<<"1\n";
	    // THIS IS THE PROBLEM:
	    matrix dh = dot(transpose(why), dy);
	    std::cout << "NROWS: " << n_rows <<"\n";
	    // backpropagate	
	    for (int idx = n_rows; idx >= 0; idx--) {
		//std::cout<<idx<<"\t";	
		// dbh += (1 - get_row(prior_hs, n_rows) ** 2) * dh
		matrix h_row_temp = get_col(prior_hs, n_rows);
		//h_row_temp = transpose(h_row_temp);
		pow_e_wise(h_row_temp, 2L);
		// this mult then add could be condensed to 1 op, scalar - matrix
		multiply_scalar(h_row_temp, -1);
		add_scalar(h_row_temp, 1);
		multiply(h_row_temp, dh);
		
		add_in_place(dbh, h_row_temp);

		matrix h_row = get_row(prior_hs, n_rows);

		add_in_place(dwhh, dot(h_row_temp, transpose(h_row)));
		
		add_in_place(dwxh, dot(h_row_temp, transpose(h_row)));
		
		//std::cout<<"2\n";
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
	
	// binary cross-entropy
	// y dim = (output_dim x 1)
	// forward output dim = (output_dim x 1)
	double loss(const matrix &x, const matrix &y) {
	    //std::cout<<"forward call\n";
	    std::tuple<matrix, matrix> result = forward(x);
	    // predictions
	    matrix out = std::get<0>(result);
	    
	    // not sure about this softmax
	    softmax_in_place(out);
	    
	    //print_matrix(out);

	    std::vector<element_type> y_ps = std::get<0>(out);
	    std::vector<element_type> y_vals = std::get<0>(y);

	    double loss = 0.0;
	    
	    //std::cout<< "this new loop\n";
	    //std::cout << "y_vals size: " << y_vals.size() <<"\n";
	    //std::cout << "y_ps size: " << y_ps.size() <<"\n";
	    for (size_t idx = 0; idx < y_vals.size(); idx++) {
		const double y_val = y_vals.at(idx);
		const double y_p = y_ps.at(idx);
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

	    return -1 * l / Y.size();
	}

	void train(const std::vector<matrix> &X, const std::vector<matrix> &Y) {
	    size_t num_epochs = 100;
	    
	    //std::vector<element_type> y_vals = std::get<0>(y);

	    for (size_t epoch = 0; epoch < num_epochs; epoch++) {
		
		std::cout << "Total loss: " << total_loss(X, Y) << "\n";	

		// for each training example
		for (size_t idx = 0; idx < Y.size(); idx++) {
		    //std::cout << "forward call\n";
		    std::tuple<matrix, matrix> forward_res = forward(X.at(idx));
		    
		    // tuple is y, h
		    matrix dldy = std::get<0>(forward_res);
		    softmax_in_place(dldy);

		    subtract_in_place(dldy, Y.at(idx));
		    //std::cout<<"backward call \n";
		    dldy = transpose(dldy);
		    backward(dldy);
		}
	    }
	}
	//double total_loss() {

	//}
};

