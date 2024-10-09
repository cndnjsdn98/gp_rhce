#ifndef _RANDOM_UTILS_H
#define _RANDOM_UTILS_H

#include <Eigen/Dense>
#include <random>

class StandardNormal {
public:
    StandardNormal() : gen(std::random_device{}()), dist(0.0, 1.0) {}

    // Fill an Eigen::VectorXd with random values from a normal distribution
    void fillWithNormal(Eigen::VectorXd& vec) {
        for (int i = 0; i < vec.size(); ++i) {
            vec(i) = dist(gen);
        }
    }

private:
    std::mt19937 gen; // Random-number generator
    std::normal_distribution<double> dist; // Normal distribution (mean = 0, stddev = 1)
};


#endif
