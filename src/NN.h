#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>       // std::chrono::system_clock
#include "linalg.h"
#include "Data.h"


template<class Type_>
std::pair<std::vector<vec<Type_>>, std::vector<vec<Type_>>> shuffled(std::vector<vec<Type_>> const& X, std::vector<vec<Type_>> const& Y) {
    std::vector<std::pair<vec<Type_>, vec<Type_>>> sfl;
    for (int i = 0; i < X.size(); ++i) {
        sfl.emplace_back(std::make_pair(X[i], Y[i]));
    }
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    auto rng = std::default_random_engine(seed);
    std::shuffle(std::begin(sfl), std::end(sfl), rng);

    std::vector<vec<Type_>> x_s;
    std::vector<vec<Type_>> y_s;
    for (int i = 0; i < sfl.size(); ++i) {
        x_s.emplace_back(sfl[i].first);
        y_s.emplace_back(sfl[i].second);
    }
    return std::make_pair(x_s, y_s);
}



class Layer;
class Optimizer;
class Network;



class Layer {
public:
    virtual vec<double> forward(vec<double> const& input) = 0;
    virtual vec<double> backward(vec<double> const& input, vec<double> const& grad_output) = 0;
    virtual void update() = 0;
};


class Optimizer {
public:
    virtual void init(int input_units, int output_units) = 0;
    virtual void accumulate(
        mat<double> const& grad_weights, 
        vec<double> const& grad_biases
    ) = 0;
    virtual void apply(
        mat<double>& weights,
        vec<double>& biases
    ) = 0;
};


class SGD : public Optimizer {
private:
    mat<double> v_weights;
    vec<double> v_biases;

    double learning_rate = 0.01;
    double b1 = 0.0;
public:
    void init(int input_units, int output_units) {
        v_weights.reshape(input_units, output_units);
        v_biases.reshape(output_units);

        v_weights.fill(0.0);
        v_biases.fill(0.0);
    }

    void accumulate(
        mat<double> const& grad_weights,
        vec<double> const& grad_biases
    ) override {
        // v_weights = (1.0 - b1) * grad_weights + b1 * v_weights
        v_weights = v_weights + b1 * (v_weights - grad_weights) + grad_weights;
        v_biases = v_biases + b1 * (v_biases - grad_biases) + grad_biases;
    }

    void apply(
        mat<double>& weights,
        vec<double>& biases
    ) override {
        weights = weights - learning_rate * v_weights;
        biases = biases - learning_rate * v_biases;

        v_weights.fill(0.0);
        v_biases.fill(0.0);
    }
};

/*
class Adam : public Optimizer {
private:
    mat<double> v_weights;
    vec<double> v_biases;

    mat<double> s_weights;
    vec<double> s_biases;

    int it = 1;

    double learning_rate = 0.001;
    double b1 = 0.9;
    double b2 = 0.999;
    double e = 1e-8;
public:
    void init(int input_units, int output_units) {
        v_weights.reshape(input_units, output_units);
        v_biases.reshape(output_units);

        s_weights.reshape(input_units, output_units);
        s_biases.reshape(output_units);
    }
    void optimize(
        mat<double>& weights, 
        vec<double>& biases, 
        mat<double> const& grad_weights,
        vec<double> const& grad_biases
    ) override {
        v_weights = b1 * v_weights + (1.0 - b1) * grad_weights;
        v_biases = b1 * v_biases + (1.0 - b1) * grad_biases;
        
        s_weights = b2 * s_weights + (1.0 - b2) * grad_weights.pow(2.0);
        s_biases = b2 * s_biases + (1.0 - b2) * grad_biases.pow(2.0);
        
        v_weights = v_weights / (1.0 - std::pow(b1, it));
        v_biases = v_biases / (1.0 - std::pow(b1, it));
        
        s_weights = s_weights / (1.0 - std::pow(b2, it));
        s_biases = s_biases / (1.0 - std::pow(b2, it));
        
        weights = weights - learning_rate * v_weights / (s_weights.pow(0.5) + e);
        biases = biases - learning_rate * v_biases / (s_biases.pow(0.5) + e);
        
        it++;
    }
};
*/


class ReLU : public Layer {
public:
    vec<double> forward(vec<double> const& input) override {
        return maximum(vec<double>::filled(input.cols()), input);
    }

    vec<double> backward(vec<double> const& input, vec<double> const& grad_output) override {
        return grad_output * input.mapped([](double& obj) {
            obj = obj > 0;
        });;
    }

    void update() {}
};


class BatchNormalization : public Layer {
public:
    vec<double> forward(vec<double> const& input) override {
        return maximum(vec<double>::filled(input.cols()), input);
    }

    vec<double> backward(vec<double> const& input, vec<double> const& grad_output) override {
        return grad_output * input.mapped([](double& obj) {
            obj = obj > 0;
            });;
    }

    void update() {}
};


class Sigmoid : public Layer {
public:
    vec<double> forward(vec<double> const& input) override {
        return input.mapped([](double& obj) {
            obj = 1.0 / (1.0 + exp(-obj));
        });
    }

    vec<double> backward(vec<double> const& input, vec<double> const& grad_output) override {
        return grad_output * input.mapped([](double& obj) {
            obj = (1.0 / (1.0 + exp(-obj))) * (1.0 - (1.0 / (1.0 + exp(-obj))));
        });
    }

    void update() {}
};


class Dense : public Layer {
private:
    mat<double> weights;
    vec<double> biases;

    Optimizer* m_optimizer = nullptr;
public:
    Dense(int input_units, int output_units, Optimizer* optimizer = new SGD()) {
        weights.reshape(input_units, output_units);
        biases.reshape(output_units);

        weights.randomize();
        biases.randomize();

        m_optimizer = optimizer;
        m_optimizer->init(input_units, output_units);
    }

    vec<double> forward(vec<double> const& input) override {
        return input * weights + biases;
    }

    vec<double> backward(vec<double> const& input, vec<double> const& grad_output) override {
        auto grad_input = grad_output * weights.transposed();
        auto grad_weights = input.transposed() * grad_output;
        auto grad_biases = grad_output;

        m_optimizer->accumulate(grad_weights, grad_biases);

        return grad_input;
    }

    void update() override {
        m_optimizer->apply(weights, biases);
    }

    ~Dense() {
        delete m_optimizer;
    }
};


class Loss {
public:
    virtual double getLoss(vec<double> const& logits, vec<double> const& r_answers) = 0;
    virtual vec<double> getLossGrad(vec<double> const& logits, vec<double> const& r_answers) = 0;
};


class LossSoftmaxCrossentropy : public Loss {
public:
    double getLoss(vec<double> const& logits, vec<double> const& r_answers) override {
        auto logits_norm = logits - logits.data()[logits.index_maximum()];
        return (
            -logits_norm.data()[r_answers.index_maximum()]
            + std::log(logits_norm.mapped([](double& obj) {
                obj = std::exp(obj);
            }).sum())
        );
    }
    vec<double> getLossGrad(vec<double> const& logits, vec<double> const& r_answers) override {
        vec<double> ones_for_answers = vec<double>::filled(logits.cols());
        ones_for_answers.data()[r_answers.index_maximum()] = 1.0;

        auto logits_norm = logits - logits.data()[logits.index_maximum()];

        vec<double> l_exp = logits_norm.mapped([](double& obj) {
            obj = std::exp(obj);
        });

        auto softmax = l_exp / l_exp.sum();
        auto out = (softmax - ones_for_answers);

        return out;
    }
};


class LossMSE : public Loss {
public:
    double getLoss(vec<double> const& logits, vec<double> const& r_answers) override {
        return (r_answers - logits).dot(r_answers - logits) / logits.cols();
    }
    vec<double> getLossGrad(vec<double> const& logits, vec<double> const& r_answers) override {
        return -2 * (r_answers - logits);
    }
};


class Network {
private:
    Loss* m_loss_func = nullptr;
    Optimizer* m_optimizer = nullptr;
    std::vector<Layer*> m_layers;

    std::vector<vec<double>> _forward(vec<double> const& X){
        std::vector<vec<double>> out(m_layers.size() + 1);
        out[0] = X;
        for(int i = 0; i < m_layers.size(); ++i)
            out[i + 1] = m_layers[i]->forward(out[i]);
        return out;
    }

public:
    Network(
        Loss* loss_func, 
        Optimizer* optimizer = new SGD()
    ) : m_loss_func(loss_func), m_optimizer(optimizer) {}
    ~Network() {
        delete m_loss_func;
        delete m_optimizer;
    }

    void addLayer(Layer* layer){
        m_layers.push_back(layer);
    }

    void train(Data<double>& data, int epochs = 10, int batches = 32){
        double loss = 0;

        double m_loss = 0;
        int m_l_count = 0;

        auto& l_data = data.getData();
        for (int ep = 0; ep < epochs; ++ep) {
            loss = 0;
            data.shuffle();

            for (int it = 0; it < batches; ++it) {
                auto A = _forward(l_data[it].first);
                vec<double> curr_backward = m_loss_func->getLossGrad(A.back(), l_data[it].second) / double(batches);
                loss += m_loss_func->getLoss(A.back(), l_data[it].second);
                for (int i = m_layers.size() - 1; i >= 0; --i)
                    curr_backward = m_layers[i]->backward(A[i], curr_backward);
            }
            for (int i = m_layers.size() - 1; i >= 0; --i)
                 m_layers[i]->update();

            loss /= batches;
            m_loss = (m_loss * m_l_count + loss)/(double)(m_l_count+1);
            m_l_count++;
            if (ep % 1000 == 0) {
                printf("Ep: %d, loss: %lf, mloss: %lf\n", ep, loss, m_loss);
                //if (m_loss <= 0.0001) break;
                m_l_count = 0;
                m_loss = 0;
            }
        }
    }

    vec<double> predict(vec<double> const& X){
        auto A = _forward(X);
        return A.back();
    }
};