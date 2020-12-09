//#include "utils.h"
#include "rnn.h"
#include "test.h"

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>

// an 'opts' tuple to make the function get_opts more readable
// opts: prot_fp, ligand_fp, ligand_actual_fp, lj_fp, eval_bool, test_bool, 
// help_bool, num_samples
typedef std::tuple<std::string, std::string, std::string, 
	std::string, std::string> opts;

// prints the help message
void print_help() {
    printf("FLAGS\n");
    printf("\t-e\tIf present, only RMSD between the input ligand and the\n");
    printf("\t\tactual (ground truth) will be calculated (no docking)\n");
    printf("\t-h\tShow help message\n");
    printf("OPTIONS\n");
    printf("\t-p <protein fp>\tProtein file path, a PDB file with atom types\n");
    printf("\t\t\tcorresponding to those in the Lennard-Jones params file\n");
    printf("\t-l <ligand fp>\tLigand file path, a PDB file with atom types\n");
    printf("\t\t\tcorresponding to those in the Lennard-Jones params file\n");
    printf("\t-a <ligand fp>\tActual (ground truth) ligand file path,\n");
    printf("\t\t\ta PDB file with atom types corresponding to those in the\n");
    printf("\t\t\tLennard-Jones params file\n");
    printf("\t-j <lj fp>\tFile path to the Lennard-Jones params file\n");
    printf("\t-n <num>\tNumber of angles to test, default: 120\n\n");
}

// parses options with GNU getopt and returns them in a tuple
// used template from 
// https://www.gnu.org/software/libc/manual/html_node/Example-of-Getopt.html
// and
// https://www.gnu.org/software/libc/manual/html_node/Getopt-Long-Option-Example.html
opts get_opts(int argc, char **argv) {
    std::string train_dir;
    std::string test_dir;
    std::string input_dim;
    std::string output_dim;
    std::string hidden_dim;

    int c;
    
    while (1) {
	static struct option long_options[] = {
	    {"train-dir", required_argument, 0, 't'},
	    {"test-dir", required_argument, 0, 's'},
	    {"input-dim", required_argument, 0, 'i'},
	    {"output-dim", required_argument, 0, 'o'},
	    {"hidden-dim", required_argument, 0, 'h'},
	    {0, 0, 0, 0}
	};

	int opt_idx = 0;
	c = getopt_long(argc, argv, "t:s:i:o:h:", long_options, &opt_idx);

	if (c == -1) {
	    break;
	}

	switch (c) {
	    case 't':
		train_dir = optarg;
		break;
	    case 's':
		test_dir = optarg;
		break;
	    case 'i':
		input_dim = optarg;
		break;
	    case 'o':
		output_dim = optarg;
		break;
	    case 'h':
		hidden_dim = optarg;
		break;
	    default:
		printf("This is a naive arg parser and is not tolerant\n");
		printf("of any uncertainty\n");
		print_help();
	}
    }
    return std::make_tuple(train_dir, test_dir, input_dim, output_dim, hidden_dim);	
}

int main(int argc, char **argv) {
    // test_routine();
    opts options = get_opts(argc, argv);
    std::string train_dir = std::get<0>(options);
    std::string test_dir = std::get<1>(options);
    size_t input_dim = std::stoul(std::get<2>(options));
    size_t output_dim = std::stoul(std::get<3>(options));
    size_t hidden_dim = std::stoul(std::get<4>(options));


    std::cout << "Model initialization\n";
    RNN model = RNN(input_dim, output_dim, hidden_dim);

    std::cout << "Loading training data\n";
    std::tuple<std::vector<matrix>, std::vector<matrix>> load_result = load_from_dir(train_dir);

    std::vector<matrix> X = std::get<0>(load_result);
    std::vector<matrix> Y = std::get<1>(load_result);

    std::cout << "Training\n";
    model.train(X, Y);

    std::cout << "Loading test data\n";

    load_result = load_from_dir(test_dir);
    X = std::get<0>(load_result);
    Y = std::get<1>(load_result);
    model.test(X, Y);
}
