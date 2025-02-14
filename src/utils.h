#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <math.h>
#include <random>
#include <regex>
#include <stdlib.h>
#include <string>
#include <tuple>
#include <vector>
namespace fs = std::filesystem;

typedef double element_type;
typedef size_t dim;
typedef std::tuple<std::vector<element_type>, dim> matrix;

std::random_device RANDOM_DEVICE;
std::mt19937 generator(RANDOM_DEVICE());

// prints the matrix, currently only used for debugging
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

// generates a random matrix appropriate for the tanh activation function
matrix gen_random_matrix(size_t m, size_t n, size_t prior_layer_dim) {
    std::vector<element_type> vals_out;

    element_type min = -1 * (1 / sqrt(prior_layer_dim));
    element_type max = 1 / sqrt(prior_layer_dim);

    std::uniform_real_distribution<element_type> distribution(min, max);

    for (size_t _idx = 0; _idx < (m * n); _idx++) {
	vals_out.push_back(distribution(generator));
    }

    return std::make_tuple(vals_out, n);
}

// generates a matrix of 0s
matrix gen_zeros_matrix(size_t m, size_t n) {
    std::vector<element_type> vals_out;

    for (size_t _idx = 0; _idx < (m * n); _idx++) {
	vals_out.push_back(0);
    }

    return std::make_tuple(vals_out, n);
}

// matrix multiplication
matrix dot(const matrix &a, const matrix &b) {
    dim a_dim = std::get<1>(a);
    std::vector<element_type> a_vals = std::get<0>(a);

    dim b_dim = std::get<1>(b);
    std::vector<element_type> b_vals = std::get<0>(b);
    dim b_n_rows = b_vals.size() / b_dim;
    // num cols & rows in output matrix
    // n_rows from a x n_cols from b
    size_t n_cols = b_dim;
    size_t n_rows = std::get<0>(a).size() / a_dim;

    std::vector<element_type> vals_out;
    vals_out.reserve(n_cols * n_rows);

    // check to make sure conformable, a_cols = b_rows
    if (a_dim != b_n_rows) {
	size_t a_0 = a_vals.size() / a_dim;
	size_t a_1 = a_dim;
	size_t b_0 = b_vals.size() / b_dim;
	size_t b_1 = b_dim;
	// this is fairly improper
	std::cout << "utils::dot - matrices are not comformable\n";
	std::cout << "matrix a: " << a_0 << " x " << a_1 << "\n";
	std::cout << "matrix b: " << b_0 << " x " << b_1 << "\n";
	exit(EXIT_FAILURE);
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

// transposes a matrix
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

// gets row i from matrix m
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

    return std::make_tuple(vec_out, m_dim);
}

// gets col j from matrix m
// returns a mx1 matrix
matrix get_col(const matrix &m, const size_t &j) {
    std::vector<element_type> vals_out;

    dim m_dim = std::get<1>(m);
    std::vector<element_type> m_vals = std::get<0>(m);

    size_t n_rows = m_vals.size() / m_dim;

    for (size_t row = 0; row < n_rows; row++) {
	vals_out.push_back(m_vals.at(row * m_dim + j));
    }

    return std::make_tuple(vals_out, 1);
}

// sigmoid activation function
void sigmoid(matrix &a) {
    std::vector<element_type> *a_vals = &std::get<0>(a);

    for (size_t idx = 0; idx < (*a_vals).size(); idx++) {
	(*a_vals).at(idx) = 1 / (1 + exp(-1 * ((*a_vals).at(idx))));
    }
}

// adds matrix b to matrix a element wise without allocating
// new memory. a is lost 
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
	std::cout << "utils::add_in_place - matrices are not same dims\n";
	std::cout << "matrix a: " << (*a_vals).size() / *a_dim << " x " << *a_dim << "\n";
	std::cout << "matrix b: " << b_vals.size() / b_dim << " x " << b_dim << "\n";
	exit(EXIT_FAILURE);
    }
}

// subtracts matrix b from matrix a element wise without allocating 
// new memory. a is lost
void subtract_in_place(matrix &a, const matrix &b) {
    dim *a_dim = &std::get<1>(a);
    std::vector<element_type> *a_vals = &std::get<0>(a);
    size_t a_n_rows = (*a_vals).size() / *a_dim;

    dim b_dim = std::get<1>(b);
    std::vector<element_type> b_vals = std::get<0>(b);
    size_t b_n_rows = std::get<0>(b).size() / b_dim;

    if (*a_dim == b_dim && a_n_rows == b_n_rows) {
	for (size_t idx = 0; idx < (*a_vals).size(); idx++) {
	    (*a_vals).at(idx) = ((*a_vals).at(idx) - b_vals.at(idx));
	}
    } else {
	std::cout << "utils::subtract_in_place - matrices are not same dims\n";
	std::cout << "matrix a: " << (*a_vals).size() / *a_dim << " x " << *a_dim << "\n";
	std::cout << "matrix b: " << b_vals.size() / b_dim << " x " << b_dim << "\n";
	exit(EXIT_FAILURE);
    }
}

// element-wise clipping for all values in m
void clip(matrix &m, element_type min, element_type max) {
    std::vector<element_type> *m_vals = &std::get<0>(m);

    for (size_t idx = 0; idx < (*m_vals).size(); idx++) {
	if ((*m_vals).at(idx) < min) {
	    (*m_vals).at(idx) = min;
	} else if ((*m_vals).at(idx) > max) {
	    (*m_vals).at(idx) = max;
	}
    }
}

// take all elements in the matrix to the power power
void pow_e_wise(matrix &m, const double power) {
    std::vector<element_type> *m_vals = &std::get<0>(m);

    for (size_t idx = 0; idx < (*m_vals).size(); idx++) {
	(*m_vals).at(idx) = powl((*m_vals).at(idx), power);
    }
}

// add a scalar to all values in the matrix
void add_scalar(matrix &m, const double scalar) {
    std::vector<element_type> *m_vals = &std::get<0>(m);

    for (size_t idx = 0; idx < (*m_vals).size(); idx++) {
	(*m_vals).at(idx) = (*m_vals).at(idx) + scalar;
    }
}

// multiply all values in the matrix m by a scalar
void multiply_scalar(matrix &m, const double scalar) {
    std::vector<element_type> *m_vals = &std::get<0>(m);

    for (size_t idx = 0; idx < (*m_vals).size(); idx++) {
	(*m_vals).at(idx) = (*m_vals).at(idx) * scalar;
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
	std::cout << "utils::multiply - matrices are not same dims\n";
	std::cout << "matrix a: " << (*a_vals).size() / *a_dim << " x " << *a_dim << "\n";
	std::cout << "matrix b: " << b_vals.size() / b_dim << " x " << b_dim << "\n";
    }
}

// append b columns to a, returning a new matrix
matrix append_cols(const matrix &a, const matrix &b) {
    dim a_dim = std::get<1>(a);
    std::vector<element_type> a_vals = std::get<0>(a);
    size_t a_n_rows = a_vals.size() / a_dim;

    dim b_dim = std::get<1>(b);
    std::vector<element_type> b_vals = std::get<0>(b);
    size_t b_n_rows = b_vals.size() / b_dim;

    std::vector<element_type> vals_out;

    if (b_n_rows == a_n_rows) {
	for (size_t row = 0; row < a_n_rows; row++) {
	    for (size_t a_idx = row * a_dim; a_idx < row * a_dim + a_dim; a_idx++) {
		vals_out.push_back(a_vals.at(a_idx));
	    }
	    for (size_t b_idx = row * b_dim; b_idx < row * b_dim + b_dim; b_idx++) {
		vals_out.push_back(b_vals.at(b_idx));
	    }
	}
    } else {
	std::cout << "utils::append_cols - a_n_rows != b_n_rows !!!\n";
    }

    return std::make_tuple(vals_out, a_dim + b_dim);
}

// append b rows to a without allocating memory for an entire 
// new matrix, a is lost
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

// round all the values in m based on a rounding threshold
void round(matrix &m, const double &threshold) {
    std::vector<element_type> *m_vals = &std::get<0>(m);

    for (size_t idx = 0; idx < (*m_vals).size(); idx++) {
	if ((*m_vals).at(idx) < threshold) {
	    (*m_vals).at(idx) = 0.0;
	} else {
	    (*m_vals).at(idx) = 1.0;
	}
    }
}

// trims whitespace from a string
std::string trim_whitespace(std::string a_string) {
    size_t first = a_string.find_first_not_of(' ');
    if (first == std::string::npos) {
	return "";
    }
    size_t last = a_string.find_last_not_of(' ');
    return a_string.substr(first, (last - first + 1));
}

// parses a single line of a comma delimited input
// file
std::vector<std::string> parse_line(std::string line) {
    std::vector<std::string> vec_out;
    std::string delim = "|";

    std::regex delim_regex(",");
    line = std::regex_replace(line, delim_regex, delim);

    size_t index = 0;

    std::string element;
    std::string trimmed_element;

    while ((index = line.find(delim)) != std::string::npos) {
	element = line.substr(0, index);

	trimmed_element = trim_whitespace(element);
	if (!trimmed_element.empty()) {
	    vec_out.push_back(trim_whitespace(element));
	}

	line.erase(0, index + delim.length());
    }

    trimmed_element = trim_whitespace(line);
    if (!trimmed_element.empty()) {
	vec_out.push_back(line);
    }

    return vec_out;
}

// loads in the matrix from a string representing a file path
std::tuple<matrix, matrix> load_sample(const std::string file_path) {
    bool dimension_known = false;
    dim dimension;
    
    std::vector<element_type> x_vals;
    std::vector<element_type> y_vals;

    std::fstream file_in;

    file_in.open(file_path, std::ios::in);

    std::string line;

    while (getline(file_in, line)) {
	std::vector<std::string> elements = parse_line(line);

	if (!dimension_known) {
	    dimension = elements.size();
	    dimension_known = true;
	} else {
	    if (elements.size() != dimension) {
		printf("Not every row has same dimension as first row\n");
		exit(EXIT_FAILURE);
	    }
	}

	for (size_t idx = 0; idx < elements.size() - 1; idx++) {
	    x_vals.push_back(std::stod(elements.at(idx)));
	}

	y_vals.push_back(std::stod(elements.at(elements.size() - 1)));
    }

    matrix x = std::make_tuple(x_vals, dimension - 1);
    matrix y = std::make_tuple(y_vals, 1);

    file_in.close();
    
    return std::make_tuple(x, y);
}

// loads dataets from the directory passed as dir_path
std::tuple<std::vector<matrix>, std::vector<matrix>> load_from_dir(const std::string dir_path) {
    std::vector<matrix> X;
    std::vector<matrix> Y;

    // std::cout << "starting load loop\n";
    for (const auto &entry : fs::directory_iterator(dir_path)) {
	std::tuple<matrix, matrix> matrices = load_sample(entry.path().string());
	X.push_back(std::get<0>(matrices));
	Y.push_back(std::get<1>(matrices));
    }

    return std::make_tuple(X, Y);
}

// write a matrix to file
void write(const matrix &m, const std::string file_path) {
    std::ofstream file_out;
    file_out.open(file_path);

    std::vector<element_type> m_vals = std::get<0>(m);
    dim m_dim = std::get<1>(m);

    size_t idx = 0;

    for (element_type val : m_vals) {
	idx++;
	if (idx != 0 && idx % m_dim == 0) {
	    file_out << val << "\n";	
	} else {
	    file_out << val << ",";
	}
    }

    file_out.close();
}

// loads a weights matrix
matrix load_weights_matrix(const std::string file_path, const dim m_dim, const dim n_dim) {
    std::vector<element_type> m_vals;
    size_t n_rows = 0;

    std::fstream file_in;
    file_in.open(file_path, std::ios::in);

    std::string line;

    while (getline(file_in, line)) {
	std::vector<std::string> elements = parse_line(line);

	if (elements.size() != n_dim) {
	    printf("utils::load_weights_matrix - bad dimension\n");
	    exit(EXIT_FAILURE);
	}

	for (std::string element : elements) {
	    m_vals.push_back(std::stod(element));
	}
	
	if (elements.size() != 0) {
	    n_rows++;
	}
    }

    if (n_rows != m_dim) {
	std::cout << "utils::load_weights_matrix - bad m dim\n";
	exit(EXIT_FAILURE);
    }

    file_in.close();

    matrix m = std::make_tuple(m_vals, n_dim);

    return m;
}

// randomly shuffle a vector of matrices
void shuffle(std::vector<matrix> matrices) {
    std::shuffle(matrices.begin(), matrices.end(), generator);
}

// A special matrix multiplication function. 
// multiplies the first matrix by the transpose
// of the second matrix. avoids a memory allocation
matrix dott(const matrix &a, const matrix &b) {
    dim a_dim = std::get<1>(a);
    std::vector<element_type> a_vals = std::get<0>(a);

    dim b_dim = std::get<1>(b);
    std::vector<element_type> b_vals = std::get<0>(b);

    std::vector<element_type> vals_out;
    vals_out.reserve(b_vals.size() / b_dim * a_vals.size() / a_dim);

    // check to make sure conformable, a_rows = b_rows
    if (a_dim != b_dim) {
	std::cout << "utils::dott - matrices are not comformable\n";
	exit(EXIT_FAILURE);

    } else {
	// is there a faster way to do this??
	for (size_t a_row = 0; a_row < a_vals.size() / a_dim; a_row++) {
	    for (size_t b_row = 0; b_row < b_vals.size() / b_dim; b_row++) {
		element_type sum = 0;
		
		size_t a_idx = a_row * a_dim;
		size_t b_idx = b_row * b_dim;

		for (size_t col = 0; col < a_dim; col++) {
		    sum += (a_vals.at(a_idx + col) * b_vals.at(b_idx + col));
		}
		
		vals_out.push_back(sum);
	    }
	}
    }
    return std::make_tuple(vals_out, b_vals.size() / b_dim);
}
