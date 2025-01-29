#include "skepu"
#include <iostream>

int add(int a, int b) { return a+b; }

auto sum() {
    static auto var = skepu::Reduce(add);
    return var;
}

int main() {
    auto summ = sum();
    skepu::Vector<int> a(10);


    for (int i = 0; i < 10; i++) {
        a(i) += i+1;
    }


    summ(a);
}