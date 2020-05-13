#include <torch/torch.h>

struct LSTMTest : torch::nn::Module
{
    LSTMTest()
    {
        torch::nn::GRUOptions opts(256, 128);
        opts.bidirectional(true);
        opts.cat_layer_fwd_bwd_states(true);
        opts.num_layers(3);
        rnn = register_module("rnn", torch::nn::GRU(opts));
        // fc1 = register_module("fc1", torch::nn::Linear(784, 64));
        // fc2 = register_module("fc2", torch::nn::Linear(64, 32));
        // fc3 = register_module("fc3", torch::nn::Linear(32, 10));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        torch::Tensor state = {};
        // torch::nn::RNNOutput output;
        std::cout << "Calling forward\n";
        auto output = rnn->forward(x);
        return std::get<0>(output);
    }

    torch::nn::GRU rnn{nullptr};
};

int main()
{
    // at::set_num_threads(1);
    at::globalContext().setUserEnabledCuDNN(false);
    torch::DeviceType device_type;
    device_type = torch::kCUDA;

    std::cout << "start?\n";
    auto net = std::make_shared<LSTMTest>();
    //net -> to(device_type);
    std::cout << "Sending model to GPU\n";
    // std::cout << net->children();
    net->pretty_print(std::cout);
    std::cout << "\n";
    // TxBxE
    torch::Tensor input = torch::rand({10, 1, 256});
    // input = input.to(device_type);
    torch::Tensor output = net->forward(input);
    std::cout << output.sizes();
    return 0;
}
