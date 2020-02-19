#include <torch/torch.h>

struct LSTMTest : torch::nn::Module {
    LSTMTest() {
        rnn = register_module("rnn", torch::nn::LSTM(256, 128));
        // fc1 = register_module("fc1", torch::nn::Linear(784, 64));
        // fc2 = register_module("fc2", torch::nn::Linear(64, 32));
        // fc3 = register_module("fc3", torch::nn::Linear(32, 10));
    }

    torch::Tensor forward(torch::Tensor x) {
        torch::Tensor state = {};
        torch::nn::RNNOutput output;
        output = rnn -> forward(x, state);
        return output.output;
    }

    torch::nn::LSTM rnn{nullptr};
};

int main() {
    auto net = std::make_shared<LSTMTest>();
    torch::Tensor input = torch::rand({1, 10, 256});
    torch::Tensor output = net -> forward(input);
    return 0;
}
