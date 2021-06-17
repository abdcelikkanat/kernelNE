//
//
//

#ifndef FAST_BERN_MODEL_H
#define FAST_BERN_MODEL_H

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <random>
#include <iomanip>
#include "math.h"
#include "Unigram.h"
#include "Vocabulary.h"
#include <algorithm>

using namespace std;

class Model {

private:
    string corpusFile, kernel;
    unsigned int dim_size, window_size, negative_sample_size, num_of_iters;
    double sigma, *kernelParams, lr, min_lr, decay_rate, lambda, beta;

    unsigned long vocab_size;
    unordered_map <string, int> node2Id;
    vector <Node> vocab_items;
    int total_nodes;
    double **emb0, **emb1;

    default_random_engine generator;
    Unigram uni;

    double sigmoid(double z);
    double gaussian_kernel(int contextId, int centerId, double sigma);
    double schoenberg_kernel(int contextId, int centerId, double sigma);
    void get_gaussian_kernel_grad(double *&g, double e, int contextId, int centerId, double sigma);

    void update_rule_nokernel(vector <double> labels, int centerId, vector <int> contextIds, double current_lr);
    void update_rule_gaussian_kernel(vector <double> labels, int centerId, vector <int> contextIds, double current_lr);
    void update_rule_schoenberg_kernel(vector <double> labels, int centerId, vector <int> contextIds, double current_lr);

    void get_gaussian_grad(double *&g, double label, double var, int centerId, int contextId, double current_lr);
    void update_gaussian_multiple_kernel(vector <double> labels, int centerId, vector <int> contextIds, double current_lr, int numOfKernels, double *kernelCoefficients);
    void get_schoenberg_grad(double *&g, double label, double alpha, int contextId, int centerId, double current_lr);
    void update_schoenberg_multiple_kernel(vector <double> labels, vector <int> contextIds, int centerId,  double current_lr, int numOfKernels, double *&kernelCoefficients);
    void update_gauss_sch_multiple_kernel(vector <double> labels, vector <int> contextIds, int centerId,  double current_lr, int numOfKernels, double *&kernelCoefficients);


    // Depreceated methods
    void inf_poly_kernel(double alpha, vector <double> labels, int centerId, vector <int> contextIds);
    void update_gaussian_multiple_kernel2(vector <double> labels, int centerId, vector <int> contextIds, double current_lr, int numOfKernels, double *kernelCoefficients);

public:

    Model(string &corpusFile, string &kernel, double *kernelParams,
          unsigned int &dimension, unsigned int &window, unsigned int &neg,
          double &lr, double &min_lr, double &decay_rate, double &lambda, double &beta, unsigned int &iter);
    ~Model();

    void run();
    void save_embeddings(string file_path);
    void save_embeddings(string file_path, int layerId);

};



#endif //FAST_BERN_MODEL_H
