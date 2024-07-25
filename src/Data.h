#pragma once
#include <vector>
#include <random>
#include "linal.h"

template<class Type_>
class Data {
private:
    std::vector<std::pair<vec<Type_>, vec<Type_>>> m_data;
public:
    Data() = default;

    void load(
        std::vector<vec<Type_>> const& X, 
        std::vector<vec<Type_>> const& Y
    ) {
        assert(X.size() == Y.size());
        m_data.clear();

        for (int i = 0; i < X.size(); ++i) 
            m_data.emplace_back(std::make_pair(X[i], Y[i]));
    }

    void shuffle(){
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        auto rng = std::default_random_engine(seed);
        std::shuffle(m_data.begin(), m_data.end(), rng);
    }

    std::vector<std::pair<vec<Type_>, vec<Type_>>>& getData() {
        return m_data;
    }
};