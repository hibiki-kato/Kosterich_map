#include <iostream>
#include <chrono>
#include <eigen3/Eigen/Dense>
#include "shared/myFunc.hpp"
#include "shared/Map.hpp"
#include "shared/Eigen_numpy_converter.hpp"
#include "shared/matplotlibcpp.h"
namespace plt = matplotlibcpp;
int main(){
    auto start = std::chrono::system_clock::now();
    KMparams params;
    params.alpha = 0.1;
    params.sigma = 0.4;
    long long n = 1e+5;
    long long dump = 1e+4;
    Eigen::VectorXd x_0 = Eigen::VectorXd::Random(2);

    KostelichMap KM(params, n, dump, x_0);
    Eigen::MatrixXd trajectory = KM.get_trajectory();
    std::vector<double> x(n+1), y(n+1);
    for (int i = 0; i < n+1; i++){
        x[i] = trajectory(0, i);
        y[i] = trajectory(1, i);
    }
    plt::figure_size(1000, 1000);
    plt::scatter(x, y);
    plt::show();
    myfunc::duration(start);
}