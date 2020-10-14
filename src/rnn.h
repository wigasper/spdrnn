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

	std::tuple<matrix, matrix> forward(matrix &x) {
	    size_t whh_n_rows = std::get<0>(whh).size() / std::get<1>(whh);
	    matrix h = gen_zeros_matrix(whh_n_rows, 1);
	    
	    std::vector<element_type> x_vals = std::get<0>(x);
	    dim x_dim = std::get<1>(x);
	    
	    size_t x_n_rows = x_vals.size() / x_dim;
	    
	    /// for backwards phase
	    prior_inputs = x;
	    
	    prior_hs = transpose(h);
	    //append_rows(prior_hs, h);

	    for (size_t row = 0; row < x_n_rows; row++) {
		// avoid memory allocation here/
		matrix x_row = get_row(x, row);

		matrix sum = dot(wxh, x_row);
		matrix t_1 = dot(whh, h);
		add_in_place(sum, t_1);
		add_in_place(sum, h);

		tanh_e_wise(sum);
		h = sum;

		/// for backwards phase
		// possibly need to figure this out, make more efficient
		append_rows(prior_hs, transpose(h));
	    }
	    
	    matrix y = dot(why, h);
	    add_in_place(y, by);
	    
	    // TODO: can return h vals as well
	    return std::make_tuple(y, h);
	}

	// dy matrix will have dim m x 1
	void backward(matrix &dy) {
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




		
	    for (size_t idx = n_rows; idx > -1; idx--) {
		
	    }
	}
};

