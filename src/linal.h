#pragma once
#include <algorithm>
#include <cassert>
#include <iostream>
#include <initializer_list>
#include <cstdlib>
#include <vector>


template<class Type_ = double>
struct mat{
    mat() {
        m_data = nullptr;
    };
    mat(int n_rows, int n_cols) : m_rows(n_rows), m_cols(n_cols) {
        reshape(n_rows, n_cols);
    }
    mat(std::initializer_list<std::initializer_list<Type_>> lst)
    : m_rows(lst.size()), m_cols(lst.begin()->size())
    {
        reshape(m_rows, m_cols);
        Type_* ptr = m_data;
        for(auto row = lst.begin(); row != lst.end(); ++row, ptr += m_cols){
            assert((*row).size() == m_cols);
            std::copy(row->begin(), row->end(), ptr);
        }
    }
    mat(mat<Type_> const& other) {
        if (&other != this) {
            this->reshape(other.m_rows, other.m_cols);
            memcpy(m_data, other.m_data, sizeof(Type_) * other.m_rows * other.m_cols);
        }
    }
    mat(mat<Type_>&& other) noexcept {
        if (m_data) delete[] m_data;

        m_cols = other.m_cols;
        m_rows = other.m_rows;
        m_data = other.m_data;

        other.m_data = nullptr;
    }
    ~mat() { if (m_data) delete[] m_data; }

    mat<Type_>& operator=(mat<Type_> const& other) {
        if (&other != this) {
            this->reshape(other.m_rows, other.m_cols);
            memcpy(m_data, other.m_data, sizeof(Type_) * other.m_rows * other.m_cols);
        }
        return *this;
    }

    mat<Type_>& operator=(mat<Type_>&& other) noexcept {
        if (m_data) delete[] m_data;

        m_cols = other.m_cols;
        m_rows = other.m_rows;
        m_data = other.m_data;

        other.m_data = nullptr;

        return *this;
    }

    mat<Type_> operator*(mat<Type_> const& other) const {
        int n_rows = other.rows();
        int n_cols = other.cols();
        assert(m_cols == n_rows);
        
        mat<Type_> out;
        out.reshape(m_rows, n_cols);

        for(int r = 0; r < m_rows; ++r)
            for(int c = 0; c < n_cols; ++c)
                for (int r1 = 0; r1 < m_cols; ++r1)
                    out.data()[r * n_cols + c] += m_data[r * m_cols + r1] * other.data()[r1 * n_cols + c];

        return out;
    }

    mat<Type_> operator/(mat<Type_> const& other) const {
        int n_rows = other.rows();
        int n_cols = other.cols();
        assert(m_cols == n_cols && m_rows == n_rows);

        mat<Type_> out;
        out.reshape(m_rows, n_cols);

        for (int r = 0; r < m_rows * m_cols; ++r)
            out.data()[r] = m_data[r] / other.data()[r];

        return out;
    }

    mat<Type_> operator-(mat<Type_> const& other){
        int n_rows = other.rows();
        int n_cols = other.cols();
        assert(m_cols == n_cols && m_rows == n_rows);
        
        mat<Type_> out;
        out.reshape(m_rows, m_cols);

        for(int r = 0; r < m_rows; ++r)
            for(int c = 0; c < m_cols; ++c)
                out.data()[r * m_cols + c] = m_data[r * m_cols + c] - other.data()[r * m_cols + c];

        return out;
    }

    void transpose(){
        if(m_rows == m_cols){
            for(int r = 0; r < m_rows; ++r)
                for(int c = r + 1; c < m_cols; ++c)
                    std::swap(m_data[r * m_cols + c], m_data[c * m_cols + r]);
        } else{
            Type_* n_data = new Type_[m_rows * m_cols]();
            for(int c = 0; c < m_cols; ++c)
                for(int r = 0; r < m_rows; ++r)
                    n_data[c * m_rows + r] = m_data[r * m_cols + c];
            delete[] m_data;
            m_data = nullptr;
            m_data = n_data;
            std::swap(m_cols, m_rows);
        }
    }

    mat<Type_> transposed() const {
        mat<Type_> n_mat = *this;
        n_mat.transpose();
        return n_mat;
    }

    void reshape(int n_rows, int n_cols){
        if (m_data) {
            if (m_rows * m_cols != n_rows * n_cols) {
                delete[] m_data;
                m_data = new Type_[n_rows * n_cols]();
            }
        }
        else {
            m_data = new Type_[n_rows * n_cols]();
        }
        m_rows = n_rows;
        m_cols = n_cols;
    }

    void fill(Type_ def = Type_()){
        for(int i = 0; i < m_rows * m_cols; ++i)
            m_data[i] = def;
    }

    static mat<Type_> filled(int rows, int cols, Type_ def = Type_()){
        mat<Type_> out;
        out.reshape(rows, cols);
        out.fill(def);
        return out;
    }

    void map(void (*func)(Type_& obj)){
        for(int i = 0; i < m_rows * m_cols; ++i){
            func(m_data[i]);
        }
    }

    mat<Type_> mapped(void (*func)(Type_& obj)) const {
        mat<Type_> out = *this;
        out.map(func);
        return out;
    }

    void randomize(){
        for(int i = 0; i < m_rows * m_cols; ++i)
            m_data[i] = 1.0 - 2.0 * (std::rand() % 1000) / 1000.0;
    }

    void print() const {
        for(int r = 0; r < m_rows; ++r){
            for(int c = 0; c < m_cols; ++c)
                printf("%lf ", m_data[r * m_cols + c]);
            printf("\n");
        }
    }

    mat<Type_> pow(double pow) {
        for (int i = 0; i < this->m_cols * this->m_rows; ++i)
            this->m_data[i] = std::pow(this->m_data[i], pow);
        return *this;
    }

    mat<Type_> pow(double pow) const {
        mat<Type_> out = *this;
        for (int i = 0; i < this->m_cols * this->m_rows; ++i)
            out.data()[i] = std::pow(this->m_data[i], pow);
        return out;
    }

    Type_*  data() const { return m_data; }  
    int     cols() const { return m_cols;   }
    int     rows() const { return m_rows;   }

public:
    int m_rows = 1;
    int m_cols = 1;
    
    Type_* m_data = nullptr;
};

template<class Type_>
mat<Type_> operator*(double cnst, mat<Type_> const& other){     
    int n_rows = other.rows();
    int n_cols = other.cols();   

    mat<Type_> out;
    out.reshape(n_rows, n_cols);
    for(int r = 0; r < n_rows; ++r)
        for(int c = 0; c < n_cols; ++c)
            out.data()[r * n_cols + c] = other.data()[r * n_cols + c] * cnst;
    return out;
}

template<class Type_>
mat<Type_> operator*(mat<Type_> const& other, double cnst){     
    return cnst * other;
}

template<class Type_>
mat<Type_> operator/(mat<Type_> const& other, double cnst) {
    mat<Type_> out;
    out.reshape(other.rows(), other.cols());
    for (int c = 0; c < other.rows() * other.cols(); ++c)
        out.data()[c] = other.data()[c] / cnst;
    return out;
}


template<class Type_>
mat<Type_>&& add_matrices(mat<Type_> && a, mat<Type_ > const& b) {
    assert(a.m_cols == b.m_cols && a.m_rows == b.m_rows);
    for (int c = 0; c < a.m_rows * a.m_cols; ++c)
        a.m_data[c] = a.m_data[c] + b.m_data[c];
    return std::move(a);
}


template<class Type_>
mat<Type_> add_matrices_copy(mat<Type_> const& a, mat<Type_> const& b) {
    assert(a.cols() == b.cols() && a.rows() == b.rows());
    mat<Type_> out;
    out.reshape(a.rows(), a.cols());

    for (int c = 0; c < a.rows() * a.cols(); ++c)
        out.data()[c] = a.data()[c] + b.data()[c];
    return out;
}

template<class Type_>
mat<Type_> operator+(mat<Type_> const& a, mat<Type_> const& b) {
    return add_matrices_copy(a, b);
}

template<class Type_>
mat<Type_>&& operator+(mat<Type_>&& a, mat<Type_>&& b) {
    return std::move(add_matrices<Type_>(std::move(a), std::move(b)));
}

template<class Type_>
mat<Type_>&& operator+(mat<Type_>&& a, mat<Type_> const& b) {
    return std::move(add_matrices<Type_>(std::move(a), b));
}

template<class Type_>
mat<Type_>&& operator+(mat<Type_> const& a, mat<Type_>&& b) {
    return std::move(add_matrices<Type_>(std::move(b), a));
}

template<class Type_>
mat<Type_> operator+(mat<Type_> const& other, double cnst) {
    mat<Type_> out;
    out.reshape(other.rows(), other.cols());
    for (int c = 0; c < other.rows() * other.cols(); ++c)
        out.data()[c] = other.data()[c] + cnst;
    return out;
}

template<class Type_>
mat<Type_>&& operator+(mat<Type_>&& other, double cnst) {
    for (int c = 0; c < other.m_rows * other.m_cols; ++c)
        other.m_data[c] = other.m_data[c] + cnst;
    return std::move(other);
}

template<class Type_>
mat<Type_> operator-(mat<Type_> const& other, double cnst) {
    return other + (-cnst);
}

template<class Type_>
mat<Type_>&& operator-(mat<Type_>&& other, double cnst) {
    return std::move(std::move(other) + (-cnst));
}

template<class Type_>
mat<Type_> maximum(mat<Type_> const& a, mat<Type_> const& b){
    assert(a.rows() == b.rows() && a.cols() == b.cols());
    mat<Type_> out;
    out.reshape(a.rows(), a.cols());

    for(int r = 0; r < a.rows(); ++r)
        for(int c = 0; c < a.cols(); ++c)
            out.data()[r * a.cols() + c] = std::max(a.data()[r * a.cols() + c], b.data()[r * a.cols() + c]);
    
    return out;
}


template<class Type_ = double>
struct vec : mat<Type_> {
    vec() : mat<Type_>() {}
    vec(int n_cols) : mat<Type_>(1, n_cols) {}
    vec(std::initializer_list<Type_> lst) : mat<Type_>({lst}) {}

    Type_ dot(vec<Type_> const& other) const {
        assert(this->m_cols == other.cols());
        
        Type_ out = Type_();
        for(int i=0; i<this->m_cols; ++i)
            out += other.data()[i] * this->m_data[i];
        return out; 
    }

    static vec<Type_> filled(int cols, Type_ def = Type_()) {
        vec<Type_> out;
        out.reshape(cols);
        out.fill(def);
        return out;
    }

    vec<Type_> operator*(vec<Type_> const& other) const {
        assert(this->m_cols == other.cols());

        vec<Type_> out;
        out.reshape(this->m_cols);

        for(int i = 0; i < this->m_cols; ++i)
            out.data()[i] = this->m_data[i] * other.data()[i];    

        return out;
    }

    vec<Type_> operator/(vec<Type_> const& other) const {
        assert(this->m_cols == other.cols());

        vec<Type_> out;
        out.reshape(this->m_cols);

        for (int i = 0; i < this->m_cols; ++i)
            out.data()[i] = this->m_data[i] / other.data()[i];

        return out;
    }


    vec<Type_> operator*(mat<Type_> const& other) const {
        int n_rows = other.rows();
        int n_cols = other.cols();
        assert(this->m_cols == n_rows);
        
        vec<Type_> out;
        out.reshape(n_cols);

        for(int r = 0; r < this->m_rows; ++r)
            for(int c = 0; c < n_cols; ++c)
                for (int r1 = 0; r1 < this->m_cols; ++r1) 
                    out.data()[r * n_cols + c] += this->m_data[r * this->m_cols + r1] * other.data()[r1 * n_cols + c];

        return out;
    }

    vec<Type_> operator+(vec<Type_> const& other) const {
        int n_cols = other.cols();
        assert(this->m_cols == n_cols);
        
        vec<Type_> out;
        out.reshape(this->m_cols);

        for(int c = 0; c < this->m_cols; ++c)
            out.data()[c] = this->m_data[c] + other.data()[c];

        return out;
    }

    vec<Type_> operator-(vec<Type_> const& other) const {
        int n_cols = other.cols();
        assert(this->m_cols == n_cols);
        
        vec<Type_> out;
        out.reshape(this->m_cols);

        for(int c = 0; c < this->m_cols; ++c)
            out.data()[c] = this->m_data[c] - other.data()[c];

        return out;
    }

    vec<Type_> mapped(void (*func)(Type_& obj)) const {
        vec<Type_> out = *this;
        out.map(func);
        return out;
    }

    vec<Type_> pow(double pow) {
        for (int i = 0; i < this->m_cols; ++i)
            this->m_data[i] = std::pow(this->m_data[i], pow);
        return *this;
    }

    vec<Type_> pow(double pow) const {
        vec<Type_> out = *this;
        for (int i = 0; i < this->m_cols * this->m_rows; ++i)
            out.data()[i] = std::pow(this->m_data[i], pow);
        return out;
    }

    Type_ sum() const {
        Type_ out = Type_();
        for (size_t i = 0; i < this->m_cols; ++i)
            out += this->m_data[i];
        return out;
    }

    size_t index_maximum() const {
        assert(this->m_cols != 0);
        size_t out = 0;
        for (size_t i = 1; i < this->m_cols; ++i)
            if (this->m_data[out] < this->m_data[i])
                out = i;
        return out;
    }

    void reshape(int n_cols) {
        (static_cast<mat<Type_>*>(this))->reshape(1, n_cols);
    }
};

template<class Type_>
vec<Type_> maximum(vec<Type_> const& a, vec<Type_> const& b){
    assert(a.cols() == b.cols());
    vec<Type_> out;
    out.reshape(a.cols());

    for(int c = 0; c < a.cols(); ++c)
        out.data()[c] = std::max(a.data()[c], b.data()[c]);
    
    return out;
}

template<class Type_>
vec<Type_> mean(std::vector<vec<Type_>> const& data) {
    vec<Type_> out;
    out.reshape(data[0].cols());
    for (int i = 0; i < data.size(); ++i)
        out = out + data[i];
    return out / (double)data.size();
}


template<class Type_>
vec<Type_> stdev(std::vector<vec<Type_>> const& data) {
    vec<Type_> out;
    vec<Type_> mean_data = mean(data);
    out.reshape(data[0].cols());
    for (int i = 0; i < data.size(); ++i) {
        auto x = (data[i] - mean_data);
        x = x * x;
        out = out + x;
    }
    return out.pow(0.5) / (double)data.size();
}

template<class Type_>
vec<Type_> operator*(double cnst, vec<Type_> const& other){     
    int n_cols = other.cols();   

    vec<Type_> out;
    out.reshape(n_cols);
    for(int c = 0; c < n_cols; ++c)
        out.data()[c] = other.data()[c] * cnst;
    return out;
}

template<class Type_>
vec<Type_> operator*(vec<Type_> const& other, double cnst){     
    return cnst * other;
}

template<class Type_>
vec<Type_> operator/(vec<Type_> const& other, double cnst) {
    int n_cols = other.cols();

    vec<Type_> out;
    out.reshape(n_cols);
    for (int c = 0; c < n_cols; ++c)
        out.data()[c] = other.data()[c] / cnst;
    return out;
}

template<class Type_>
vec<Type_> operator+(vec<Type_> const& other, double cnst) {
    vec<Type_> out;
    out.reshape(other.cols());
    for (int c = 0; c < other.cols(); ++c)
        out.data()[c] = other.data()[c] + cnst;
    return out;
}

template<class Type_>
vec<Type_> operator-(vec<Type_> const& other, double cnst) {
    vec<Type_> out;
    out.reshape(other.cols());
    for (int c = 0; c < other.cols(); ++c)
        out.data()[c] = other.data()[c] - cnst;
    return out;
}