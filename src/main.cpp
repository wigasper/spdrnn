//#include "utils.h"
#include "rnn.h"
#include "test.h"

int main(int argc, char **argv) {
    test_routine();

    std::cout << "model init\n";
    RNN model = RNN(1, 1, 64);

    std::cout<<"loading\n";
    std::tuple<std::vector<matrix>, std::vector<matrix>> load_result =
	load_from_dir("mock_data/train");

    std::vector<matrix> X = std::get<0>(load_result);
    std::vector<matrix> Y = std::get<1>(load_result);
    
    std::cout<<"training\n";
    model.train(X, Y);
}
