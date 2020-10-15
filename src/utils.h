#include <vector>
#include <iostream>
#include <stdlib.h>
#include <tuple>
#include <math.h>

typedef double element_type;
typedef size_t dim;
typedef std::tuple<std::vector<element_type>, dim> matrix;

matrix gen_random_matrix(size_t m, size_t n) {
    std::vector<element_type> vals_out;

    for (size_t _idx = 0; _idx < (m * n); _idx++) {
	element_type val = (rand() % 100 + 1) / 3013;
	vals_out.push_back(val);
    }

    return std::make_tuple(vals_out, n);
}

matrix gen_zeros_matrix(size_t m, size_t n) {
    std::vector<element_type> vals_out;

    for (size_t _idx = 0; _idx < (m * n); _idx++) {
	vals_out.push_back(0);
    }

    return std::make_tuple(vals_out, n);
}

matrix dot(const matrix &a, const matrix &b) {
    dim a_dim = std::get<1>(a);
    std::vector<element_type> a_vals = std::get<0>(a);

    dim b_dim = std::get<1>(b);
    std::vector<element_type> b_vals = std::get<0>(b);
     
    // num cols & rows in output matrix
    // n_rows from a x n_cols from b
    size_t n_cols = b_dim;
    size_t n_rows = std::get<0>(a).size() / a_dim;
    
    std::vector<element_type> vals_out;
    vals_out.reserve(n_cols * n_rows);
    
    // check to make sure conformable, a_cols = b_rows
    if (a_dim != std::get<0>(b).size() / b_dim) {
	// this is fairly improper
	std::cout << "utils::dot - matrices are not comformable\n";
    } else {
	// is there a faster way to do this??
	for (size_t row = 0; row < n_rows; row++) {
	    for (size_t col = 0; col < n_cols; col++) {
		element_type sum = 0;
		// keep track of a and b important indices
		size_t a_begin = row * a_dim;
		size_t a_end = (row * a_dim) + a_dim;

		size_t b_idx = col;
		
		for (size_t a_idx = a_begin; a_idx < a_end; a_idx++) {
		    sum += (a_vals.at(a_idx) * b_vals.at(b_idx));
		    b_idx += b_dim;
		}
		
		vals_out.push_back(sum);
	    }   
	}
    }
    return std::make_tuple(vals_out, n_cols);
}

// element-wise tanh
void tanh_e_wise(matrix &a) {
    std::vector<element_type> *a_vals = &std::get<0>(a);

    for (size_t idx = 0; idx < (*a_vals).size(); idx++) {
	(*a_vals).at(idx) = tanh((*a_vals).at(idx));
    }
}

void print_matrix(const matrix &m) {
    std::vector<element_type> vals = std::get<0>(m);
    dim dimension = std::get<1>(m);
    
    size_t idx = 0;

    for (element_type val : vals) {
	std::cout << val << " ";
	idx++;

	if (idx % dimension == 0) {
	    std::cout << "\n";
	}
    }
    std::cout << "\n";
}

matrix transpose(matrix &m) {
    std::vector<element_type> m_vals = std::get<0>(m);
    dim m_dim = std::get<1>(m);

    std::vector<element_type> vals_out;

    size_t n_rows = m_vals.size() / m_dim;

    for (size_t col = 0; col < m_dim; col++) {
	for (size_t row = 0; row < n_rows; row++) {
	    vals_out.push_back(m_vals.at(row * m_dim + col));
	}
    }

    return std::make_tuple(vals_out, n_rows);
}

// returns a 1xn matrix
matrix get_row(const matrix &m, const size_t &i) {
    std::vector<element_type> vec_out;

    dim m_dim = std::get<1>(m);
    std::vector<element_type> m_vals = std::get<0>(m);

    auto iter_begin = m_vals.begin() + (i * m_dim);
    auto iter_end = m_vals.begin() + (i * m_dim) + m_dim;

    for (auto iter = iter_begin; iter != iter_end; ++iter) {
	vec_out.push_back(*iter);
    }

    return std::make_tuple(vec_out, 1);
}

// returns a mx1 matrix
matrix get_col(const matrix &m, const size_t &j) {
    std::vector<element_type> vals_out;

    dim m_dim = std::get<1>(m);
    std::vector<element_type> m_vals = std::get<0>(m);
    
    size_t n_rows = m_vals.size() / m_dim;

    for (size_t row = 0; row < n_rows; row++) {
	vals_out.push_back(row * m_dim + j);
    }

    return std::make_tuple(vals_out, 1);
}

void softmax_in_place(matrix &a) {
    std::vector<element_type> *a_vals = &std::get<0>(a);

    for (size_t idx = 0; idx < (*a_vals).size(); idx++) {
	(*a_vals).at(idx) = exp((*a_vals).at(idx));
    }

    double sum = 0.0;

    for (size_t idx = 0; idx < (*a_vals).size(); idx++) {
	sum += (*a_vals).at(idx);
    }

    for (size_t idx = 0; idx < (*a_vals).size(); idx++) {
	(*a_vals).at(idx) = (*a_vals).at(idx) / sum;
    }

}

void add_in_place(matrix &a, const matrix &b) {
    dim *a_dim = &std::get<1>(a);
    std::vector<element_type> *a_vals = &std::get<0>(a);
    size_t a_n_rows = (*a_vals).size() / *a_dim;

    dim b_dim = std::get<1>(b);
    std::vector<element_type> b_vals = std::get<0>(b);
    size_t b_n_rows = std::get<0>(b).size() / b_dim; 
    
    if (*a_dim == b_dim && a_n_rows == b_n_rows) {
	for (size_t idx = 0; idx < (*a_vals).size(); idx++) {
	    (*a_vals).at(idx) = ((*a_vals).at(idx) + b_vals.at(idx));
	}	    
    } else {
	// TODO: exception
    }
}

// NOTE: currently unused. could be in place
matrix subtract(const matrix &a, const matrix &b) {
    dim a_dim = std::get<1>(a);
    std::vector<element_type> a_vals = std::get<0>(a);
    size_t a_n_rows = std::get<0>(a).size() / a_dim;

    dim b_dim = std::get<1>(b);
    std::vector<element_type> b_vals = std::get<0>(b);
    size_t b_n_rows = std::get<0>(b).size() / b_dim; 
    
    std::vector<element_type> vals_out;
    vals_out.reserve(b_dim * b_n_rows);

    if (a_dim == b_dim && a_n_rows == b_n_rows) {
	for (size_t idx = 0; idx < a_vals.size(); idx++) {
	    vals_out.push_back(a_vals[idx] - b_vals[idx]);
	}	    
    } else {
	// TODO: exception
    }

    // num cols & rows in output matrix
    size_t n_cols = b_dim;
    size_t n_rows = std::get<0>(a).size() / a_dim;
    
    return std::make_tuple(vals_out, a_dim); 
}

void pow_e_wise(matrix &m, const double power) {
    std::vector<element_type> *m_vals = &std::get<0>(m);

    for (size_t idx = 0; idx < (*m_vals).size(); idx++) {
	(*m_vals).at(idx) = powl((*m_vals).at(idx), power);
    }
}

void add_e_wise(matrix &m, const double term) {
    std::vector<element_type> *m_vals = &std::get<0>(m);

    for (size_t idx = 0; idx < (*m_vals).size(); idx++) {
	(*m_vals).at(idx) = (*m_vals).at(idx) + term;
    }
}

// element-wise multiplication applied to the first
void multiply(matrix &a, const matrix &b) {
    dim *a_dim = &std::get<1>(a);
    std::vector<element_type> *a_vals = &std::get<0>(a);
    size_t a_n_rows = (*a_vals).size() / *a_dim;

    dim b_dim = std::get<1>(b);
    std::vector<element_type> b_vals = std::get<0>(b);
    size_t b_n_rows = b_vals.size() / b_dim;
    
    if (a_n_rows == b_n_rows && *a_dim == b_dim) {
	for (size_t idx = 0; idx < (*a_vals).size(); idx++) {
	    (*a_vals).at(idx) = (*a_vals).at(idx) * b_vals.at(idx);
	}
    } else {
	std::cout << "big problem";
    }
}

matrix append_cols(matrix &a, const matrix &b) {
    dim a_dim = std::get<1>(a);
    std::vector<element_type> a_vals = std::get<0>(a);
    size_t a_n_rows = a_vals.size() / a_dim;

    dim b_dim = std::get<1>(b);
    std::vector<element_type> b_vals = std::get<0>(b);
    size_t b_n_rows = b_vals.size() / b_dim;
    
    std::vector<element_type> vals_out;

    for (size_t row = 0; row < a_n_rows; row++) {
	for (size_t a_idx = row * a_dim; a_idx < row * a_dim + a_dim; a_idx++) {
	    vals_out.push_back(a_vals.at(a_idx));
	}
	for (size_t b_idx = row * b_dim; b_idx < row * b_dim + b_dim; b_idx++) {
	    vals_out.push_back(b_vals.at(b_idx));
	}
    }

    return std::make_tuple(vals_out, a_dim + b_dim);
    /* sidebarring this - probably possible but all these inserts
     * probably make it not even worth it
    *a_dim += b_dim;

    if (a_n_rows == b_n_rows) {
	for (size_t row = 0; row < a_n_rows; row++) {
	    // this is the idx of the end of the row
	    size_t idx = ((row + 1) * *a_dim) - 1;

	    auto iter_begin = (*a_vals).begin() + idx;
	    auto iter_end = (*a_vals).begin() + idx + b_dim;
	    for (auto 
	}

    } else {
	std::cout<< "utils::append_cols - a_n_rows!=b_n_rows!!!!!\n";
    }*/

}

// used to append rows
void append_rows(matrix &a, const matrix &b) {
    dim a_dim = std::get<1>(a);
    std::vector<element_type> *a_vals = &std::get<0>(a);

    dim b_dim = std::get<1>(b);
    std::vector<element_type> b_vals = std::get<0>(b);

    if (a_dim == b_dim) {
	for (element_type val : b_vals) {
	    (*a_vals).push_back(val);
	}
    } else {
	std::cout << "utils::append_matrix - a_dim != b_dim!!!!";
    }


}
