//#include "utils.h"
#include "rnn.h"
#include "test.h"

#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>

// an 'opts' tuple to make the function get_opts more readable
// TODO: intention kind of backfired and this is not so cool
// make it a map later
typedef std::tuple<std::string, std::string, std::string, std::string, std::string, std::string,
		   std::string, std::string>
    opts;

// prints the help message
void print_help() {
    printf("OPTIONS\n");
    printf("\t--train-dir\tPath to directory containing training data, where\n");
    printf("\t\t\teach file in the directory contains a sample\n");
    printf("\t--test-dir\tPath to directory containing test data, where\n");
    printf("\t\t\teach file in the directory contains a sample. Use this\n");
    printf("\t\t\toption without --train-dir to test only\n");
    printf("\t--input-dim\tDimensionality of the input layer\n");
    printf("\t--hidden-dim\tDimensionality of the hidden layer\n");
    printf("\t--output-dim\tDimensionality of the output layer\n");
    printf("\t--epochs\tNumber of epochs to train for, default: 30\n");
    printf("\t--learning-rate\tInitial learning rate\n");
    printf("\t--bptt-stop\tNumber of time steps to to go backwards during\n");
    printf("\t\t\tbackpropagation at each time step\n");
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
    std::string epochs = "30";
    std::string learning_rate = "0.0001";
    std::string bptt_stop = "20";

    int c;

    while (1) {
	static struct option long_options[] = {
	    {"train-dir", required_argument, 0, 't'},	  {"test-dir", required_argument, 0, 's'},
	    {"input-dim", required_argument, 0, 'i'},	  {"output-dim", required_argument, 0, 'o'},
	    {"hidden-dim", required_argument, 0, 'd'},	  {"epochs", required_argument, 0, 'e'},
	    {"learning-rate", required_argument, 0, 'l'}, {"bptt-stop", required_argument, 0, 'b'}, 
	    {0, 0, 0, 0}};

	int opt_idx = 0;
	c = getopt_long(argc, argv, "ht:s:i:o:d:e:l:", long_options, &opt_idx);

	if (c == -1) {
	    break;
	}

	switch (c) {
	case 'h':
	    print_help();
	    exit(EXIT_SUCCESS);
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
	case 'd':
	    hidden_dim = optarg;
	    break;
	case 'e':
	    epochs = optarg;
	    break;
	case 'l':
	    learning_rate = optarg;
	    break;
	case 'b':
	    bptt_stop = optarg;
	    break;
	default:
	    printf("This is a naive arg parser and is not tolerant\n");
	    printf("of any uncertainty\n");
	    print_help();
	}
    }
    return std::make_tuple(train_dir, test_dir, input_dim, output_dim, hidden_dim, epochs,
			   learning_rate, bptt_stop);
}

void print_opts(opts &options) {
    std::cout << "Train data dir.: " << std::get<0>(options) << "\n";
    std::cout << "Test data dir.: " << std::get<1>(options) << "\n";
    std::cout << "Input layer dim.: " << std::get<2>(options) << "\n";
    std::cout << "Output layer dim.: " << std::get<3>(options) << "\n";
    std::cout << "Hidden layer dim.: " << std::get<4>(options) << "\n";
    std::cout << "Num. epochs: " << std::get<5>(options) << "\n";
    std::cout << "Learning rate: " << std::get<6>(options) << "\n";
    std::cout << "BPTT stop: " << std::get<7>(options) << "\n";
}

int main(int argc, char **argv) {
    // test_routine();
    opts options = get_opts(argc, argv);
    std::string train_dir = std::get<0>(options);
    std::string test_dir = std::get<1>(options);
    size_t input_dim = std::stoul(std::get<2>(options));
    size_t output_dim = std::stoul(std::get<3>(options));
    size_t hidden_dim = std::stoul(std::get<4>(options));
    size_t epochs = std::stoul(std::get<5>(options));
    double learning_rate = std::stod(std::get<6>(options));
    size_t bptt_stop = std::stoul(std::get<7>(options));

    std::tuple<std::vector<matrix>, std::vector<matrix>> load_result;
    std::vector<matrix> X;
    std::vector<matrix> Y;

    print_opts(options);

    RNN model = RNN(input_dim, output_dim, hidden_dim, bptt_stop);
    
    if (!train_dir.empty()) {
	std::cout << "Loading training data\n";
	load_result = load_from_dir(train_dir);
    

	X = std::get<0>(load_result);
	Y = std::get<1>(load_result);

	std::cout << "Training\n";
	model.train(X, Y, epochs, learning_rate);
    }
    
    if (!test_dir.empty()) {
	std::cout << "Loading test data\n";

	load_result = load_from_dir(test_dir);
	X = std::get<0>(load_result);
	Y = std::get<1>(load_result);
	model.test(X, Y);
    }
}
