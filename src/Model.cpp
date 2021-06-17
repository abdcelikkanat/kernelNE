#include "Model.h"
//#include "lp_lib.h"

Model::Model(string &corpusFile, string &kernel, double *kernelParams,
             unsigned int &dimension, unsigned int &window, unsigned int &neg,
             double &lr, double &min_lr, double &decay_rate, double &lambda, double &beta, unsigned int &iter) {

    this->corpusFile = corpusFile;
    this->kernel = kernel;
    this->sigma = kernelParams[1];
    this->kernelParams = kernelParams;
    this->dim_size = dimension;
    this->window_size = window;
    this->negative_sample_size = neg;

    this->lr = lr;
    this->min_lr = min_lr;
    this->decay_rate = decay_rate;
    this->lambda = lambda;
    this->beta = beta;
    this->num_of_iters = iter;


    Vocabulary vocab(this->corpusFile);
    node2Id = vocab.getNode2Id();
    total_nodes = vocab.getTotalNodes();
    vocab_size = (int)vocab.getVocabSize();
    vocab_items = vocab.getVocabItems();


    // Set up sampling class
    vector <int> counts = vocab.getNodeCounts();
    uni = Unigram(vocab_size, counts, 0.75);

    emb0 = new double*[vocab_size];
    emb1 = new double*[vocab_size];
    for(int i=0; i<vocab_size; i++) {
        emb0[i] = new double[dim_size];
        emb1[i] = new double[dim_size];
    }


}

Model::~Model() {

    for(int i=0; i<vocab_size; i++) {
        delete [] emb0[i];
        delete [] emb1[i];
    }
    delete emb0;
    delete emb1;

}

double Model::sigmoid(double z) {

    if(z > 10)
        return 1.0;
    else if(z < -10)
        return 0.0;
    else
        return 1.0 / ( 1.0 +  exp(-z));

}

double Model::gaussian_kernel(int contextId, int centerId, double sigma) {

    double sq, var;
    auto *diff = new double[this->dim_size];

    // Compute sigma square
    var = sigma * sigma;

    for (int d = 0; d < this->dim_size; d++)
        diff[d] = this->emb1[contextId][d] - this->emb0[centerId][d];

    sq = 0.0;
    for (int d = 0; d < this->dim_size; d++)
        sq += diff[d]*diff[d];

    delete [] diff;

    return exp( -sq/(2.0*var) );
}

double Model::schoenberg_kernel(int contextId, int centerId, double sigma) {

    double inv, sum;
    auto *diff = new double[this->dim_size];

    for (int d = 0; d < this->dim_size; d++)
        diff[d] = this->emb1[contextId][d] - this->emb0[centerId][d];

    sum = 0.0;
    for(int d = 0; d < this->dim_size; d++)
        sum += diff[d] * diff[d];
    inv = 1.0 / ( 1.0 + sum );

    delete [] diff;

    return pow(inv, sigma);
}

void Model::update_rule_nokernel(vector <double> labels, int centerId, vector <int> contextIds, double current_lr) {

    double *neule;
    double *z, *g, eta, *diff;
    double e;

    neule = new double[this->dim_size];
    diff = new double[this->dim_size];
    z = new double[this->dim_size];
    g = new double[this->dim_size];

    for (int d = 0; d < this->dim_size; d++) {
        neule[d] = 0.0;
        diff[d] = 0.0;
    }

    for(int i = 0; i < contextIds.size(); i++) {

        for (int d = 0; d < this->dim_size; d++)
            diff[d] = this->emb1[contextIds[i]][d] - this->emb0[centerId][d];

        eta = 0.0;
        for (int d = 0; d < this->dim_size; d++)
            eta += this->emb1[contextIds[i]][d] * this->emb0[centerId][d];

        for (int d = 0; d < this->dim_size; d++)
            z[d] = 2.0 * ( labels[i]-eta  );

        for (int d = 0; d < this->dim_size; d++)
            g[d] = -current_lr * z[d]; // minus comes from the objective function, minimization

        for (int d = 0; d < this->dim_size; d++) {
            neule[d] += g[d]*this->emb1[contextIds[i]][d];
        }

        for (int d = 0; d < this->dim_size; d++)
            this->emb1[contextIds[i]][d] += -g[d]*this->emb0[centerId][d] - current_lr*this->lambda*(this->emb1[contextIds[i]][d]);
    }
    for (int d = 0; d < this->dim_size; d++)
        this->emb0[centerId][d] += -neule[d] - current_lr*this->lambda*(this->emb0[centerId][d]);


    delete[] neule;
    delete [] diff;
    delete [] z;
    delete [] g;
}

void Model::update_rule_gaussian_kernel(vector <double> labels, int centerId, vector <int> contextIds, double current_lr ) {

    double *neule;
    double *z, *g, eta, *diff;
    double e;
    double var = this->sigma * this->sigma;

    neule = new double[this->dim_size];
    diff = new double[this->dim_size];
    z = new double[this->dim_size];
    g = new double[this->dim_size];

    for (int d = 0; d < this->dim_size; d++) {
        neule[d] = 0.0;
        diff[d] = 0.0;
    }

    for(int i = 0; i < contextIds.size(); i++) {

        for (int d = 0; d < this->dim_size; d++)
            diff[d] = this->emb1[contextIds[i]][d] - this->emb0[centerId][d];

        eta = 0.0;
        for (int d = 0; d < this->dim_size; d++)
            eta += diff[d]*diff[d];

        /*
        if(labels[i] == 1) {
            e = exp( -eta/(2.0*var) );
            for (int d = 0; d < this->dim_size; d++)
                z[d] = 2.0 * ( 1-e  ) * ( e ) * ( diff[d]/var );
                //z[d] = 2.0 * ( 1-e  ) * ( -e ) * ( - diff[d]/var );
        } else {
            e = exp( -eta/var );
            for (int d = 0; d < this->dim_size; d++)
                z[d] = 2.0 * e * ( -diff[d]/var );
        }
        */
        e = exp( -eta/(2.0*var) );
        for (int d = 0; d < this->dim_size; d++)
            z[d] = 2.0 * ( labels[i]-e  ) * ( e ) * ( diff[d]/var );
        ///////

        for (int d = 0; d < this->dim_size; d++)
            g[d] = -current_lr * z[d]; // minus comes from the objective function, minimization

        for (int d = 0; d < this->dim_size; d++) {
            neule[d] += g[d]; /////////////
        }

        for (int d = 0; d < this->dim_size; d++)
            this->emb1[contextIds[i]][d] += g[d] - current_lr*this->lambda*(this->emb1[contextIds[i]][d]);
    }
    for (int d = 0; d < this->dim_size; d++)
        this->emb0[centerId][d] += -neule[d] - current_lr*this->lambda*(this->emb0[centerId][d]);



    delete[] neule;
    delete [] diff;
    delete [] z;
    delete [] g;
}

void Model::update_rule_schoenberg_kernel(vector <double> labels, int centerId, vector <int> contextIds, double current_lr) {

    double *neule;
    double *z, *g, eta, *diff;
    double e, scalar_val;
    double var = this->sigma * this->sigma;

    neule = new double[this->dim_size];
    diff = new double[this->dim_size];
    z = new double[this->dim_size];
    g = new double[this->dim_size];

    for (int d = 0; d < this->dim_size; d++) {
        neule[d] = 0.0;
        diff[d] = 0.0;
    }

    for(int i = 0; i < contextIds.size(); i++) {

        for (int d = 0; d < this->dim_size; d++)
            diff[d] = this->emb1[contextIds[i]][d] - this->emb0[centerId][d]; // (x-y)

        eta = 0.0;
        for (int d = 0; d < this->dim_size; d++)
            eta += diff[d]*diff[d];
        eta = eta + 1.0;
        eta = 1.0 / eta; // eta = 1 / (1 + (x-y)^2 )

        e = pow(eta, this->sigma); // ( 1 + (x-y) )^{-\alpha}
        scalar_val = 2.0 * ( labels[i] - e  ) * ( this->sigma * e * eta ) * ( 2.0 );
        for (int d = 0; d < this->dim_size; d++)
            z[d] =  scalar_val * ( diff[d] );

        for (int d = 0; d < this->dim_size; d++)
            g[d] = -current_lr * z[d]; // minus comes from the objective function, minimization

        for (int d = 0; d < this->dim_size; d++) {
            neule[d] += g[d];
        }

        for (int d = 0; d < this->dim_size; d++)
            this->emb1[contextIds[i]][d] += g[d] - current_lr*this->lambda*(this->emb1[contextIds[i]][d]);
    }
    for (int d = 0; d < this->dim_size; d++)
        this->emb0[centerId][d] += -neule[d] - current_lr*this->lambda*(this->emb0[centerId][d]); // minus comes from gradient



    delete[] neule;
    delete [] diff;
    delete [] z;
    delete [] g;
}

void Model::update_gaussian_multiple_kernel(vector <double> labels, int centerId, vector <int> contextIds, double current_lr, int numOfKernels, double *kernelCoefficients) {

    //for(int k=0; k<numOfKernels; k++)
    //    kernelCoefficients[k] = 1.0/numOfKernels;

    /* ----------- Update embedding vectors ----------- */
    double *neule;
    double *g, *temp_g;
    double eta, e, *diff,  *z;
    double var;
    double *e_values;

    neule = new double[this->dim_size]{0};
    g = new double[this->dim_size]{0};
    temp_g = new double[this->dim_size]{0};
    z = new double[this->dim_size]{0};
    diff = new double[this->dim_size]{0};
    e_values = new double[this->dim_size]{0};
    double e_values_sum=0;
    double e_values_sum_ext=0;
    double f;

    for(int i = 0; i < contextIds.size(); i++) {

        for (int d = 0; d < this->dim_size; d++)
            diff[d] = this->emb1[contextIds[i]][d] - this->emb0[centerId][d];

        eta = 0.0;
        for (int d = 0; d < this->dim_size; d++)
            eta += diff[d]*diff[d];

        e_values_sum = 0;
        e_values_sum_ext = 0;
        for(int k=0; k < numOfKernels; k++) {
            var = this->kernelParams[k+1] * this->kernelParams[k+1];
            e_values[k] = kernelCoefficients[k] * exp( -eta/(2.0*var) );
            e_values_sum += e_values[k];
            e_values_sum_ext += e_values[k] * ( 1.0 / var );
        }

        f = 2.0 * ( labels[i]-e_values_sum  ) * ( e_values_sum_ext );
        for (int d = 0; d < this->dim_size; d++)
            z[d] = f * ( diff[d] );

        for (int d = 0; d < this->dim_size; d++)
            g[d] = -current_lr * z[d]; // minus comes from the objective function, minimization


        for (int d = 0; d < this->dim_size; d++)
            neule[d] += g[d];

        for (int d = 0; d < this->dim_size; d++)
            this->emb1[contextIds[i]][d] += g[d] - current_lr*this->lambda*(this->emb1[contextIds[i]][d]);
    }
    for (int d = 0; d < this->dim_size; d++)
        this->emb0[centerId][d] += -neule[d] - current_lr*this->lambda*(this->emb0[centerId][d]);


    delete[] neule;
    delete [] g;
    delete [] temp_g;
    delete [] diff;
    delete [] z;
    delete [] e_values;
    /* ------------------------------------------------ */
    /* ----------- Update embedding vectors ----------- */
//    neule = new double[this->dim_size]{0};
//    g = new double[this->dim_size]{0};
//    temp_g = new double[this->dim_size]{0};
//
//
//    for(int i = 0; i < contextIds.size(); i++) {
//
//        for(int k=0; k < numOfKernels; k++) {
//
//            this->get_gaussian_grad(temp_g, labels[i], this->kernelParams[k+1], centerId, contextIds[i], current_lr);
//
//            for (int d = 0; d < this->dim_size; d++)
//                g[d] += (kernelCoefficients[k]) * temp_g[d]; // g[d] += (1.0 / numOfKernels) * temp_g[d];
//
//        }
//
//        for (int d = 0; d < this->dim_size; d++)
//            neule[d] += g[d];
//
//        for (int d = 0; d < this->dim_size; d++)
//            this->emb1[contextIds[i]][d] += g[d] - current_lr*this->lambda*(this->emb1[contextIds[i]][d]);
//    }
//    for (int d = 0; d < this->dim_size; d++)
//        this->emb0[centerId][d] += -neule[d] - current_lr*this->lambda*(this->emb0[centerId][d]);
//
//
//    delete[] neule;
//    delete [] g;
//    delete [] temp_g;
    /* ------------------------------------------------ */

    /* ---------- Update kernel coefficients ---------- */
    double kernelSum;
    double *ker, *totalKer;

    ker = new double[numOfKernels]{0};
    totalKer = new double[numOfKernels]{0};

    for(int i = 0; i < contextIds.size(); i++) {
        kernelSum = 0;
        for (int k = 0; k < numOfKernels; k++) {
            ker[k] = this->gaussian_kernel(contextIds[i], centerId, this->kernelParams[k + 1]);
            kernelSum += kernelCoefficients[k] * ker[k];
        }
        for (int k = 0; k < numOfKernels; k++)
            totalKer[k] += 2.0 * ( labels[i] - kernelSum ) * -ker[k];
    }

    for (int k = 0; k < numOfKernels; k++) {
        kernelCoefficients[k] += -current_lr * totalKer[k] - current_lr * this->beta * kernelCoefficients[k];
        //cout << kernelCoefficients[k] << " ";
    }
    //cout << endl;
    delete[] ker;
    /* ------------------------------------------------ */

}

void Model::get_gaussian_grad(double *&g, double label, double sigma, int centerId, int contextId, double current_lr) {

    double eta, e, *diff,  *z;
    z = new double[this->dim_size];
    diff = new double[this->dim_size];

    double var = sigma * sigma;

    for (int d = 0; d < this->dim_size; d++)
        diff[d] = this->emb1[contextId][d] - this->emb0[centerId][d];

    eta = 0.0;
    for (int d = 0; d < this->dim_size; d++)
        eta += diff[d]*diff[d];

    e = exp( -eta/(2.0*var) );
    for (int d = 0; d < this->dim_size; d++)
        z[d] = 2.0 * ( label-e  ) * ( e ) * ( diff[d]/var );

    for (int d = 0; d < this->dim_size; d++)
        g[d] = -current_lr * z[d]; // minus comes from the objective function, minimization

    delete [] diff;
    delete [] z;

}

void Model::update_schoenberg_multiple_kernel(vector <double> labels, vector <int> contextIds, int centerId, double current_lr, int numOfKernels, double *&kernelCoefficients) {

    /* ----------- Update embedding vectors ----------- */
    double *neule;
    double *g, *temp_g;
    double eta, e, *diff,  *z;
    double var;
    double *e_values;

    neule = new double[this->dim_size]{0};
    g = new double[this->dim_size]{0};
    temp_g = new double[this->dim_size]{0};
    z = new double[this->dim_size]{0};
    diff = new double[this->dim_size]{0};
    e_values = new double[this->dim_size]{0};
    double e_values_sum=0;
    double e_values_sum_ext=0;
    double f;

    for(int i = 0; i < contextIds.size(); i++) {

        for (int d = 0; d < this->dim_size; d++)
            diff[d] = this->emb1[contextIds[i]][d] - this->emb0[centerId][d];

        eta = 0.0;
        for (int d = 0; d < this->dim_size; d++)
            eta += diff[d]*diff[d];
        eta = 1.0 / ( 1.0 + eta ); // eta = (1 + (x-y)^2 ) ^ {-1}

        e_values_sum = 0;
        e_values_sum_ext = 0;
        for(int k=0; k < numOfKernels; k++) {
            e = pow(eta, this->kernelParams[k+1]); // ( 1 + (x-y) )^{-\alpha}

            e_values[k] = kernelCoefficients[k] * e;
            e_values_sum += e_values[k];
            e_values_sum_ext += e_values[k] * eta * ( 2.0 );
        }

        f = 2.0 * ( labels[i] - e_values_sum  ) * ( e_values_sum_ext );
        for (int d = 0; d < this->dim_size; d++)
            z[d] = f * ( diff[d] );

        for (int d = 0; d < this->dim_size; d++)
            g[d] = -current_lr * z[d]; // minus comes from the objective function, minimization

        for (int d = 0; d < this->dim_size; d++)
            neule[d] += g[d];

        for (int d = 0; d < this->dim_size; d++)
            this->emb1[contextIds[i]][d] += g[d] - current_lr*this->lambda*(this->emb1[contextIds[i]][d]);
    }
    for (int d = 0; d < this->dim_size; d++)
        this->emb0[centerId][d] += -neule[d] - current_lr*this->lambda*(this->emb0[centerId][d]);


    delete[] neule;
    delete [] g;
    delete [] temp_g;
    delete [] diff;
    delete [] z;
    delete [] e_values;
    /* ------------------------------------------------ */

    /* ---------- Update kernel coefficients ---------- */
    double kernelSum;
    double *ker, *totalKer;

    ker = new double[numOfKernels]{0};
    totalKer = new double[numOfKernels]{0};

    for(int i = 0; i < contextIds.size(); i++) {
        kernelSum = 0;
        for (int k = 0; k < numOfKernels; k++) {
            ker[k] = this->schoenberg_kernel(contextIds[i], centerId, this->kernelParams[k + 1]);
            kernelSum += kernelCoefficients[k] * ker[k];
        }
        for (int k = 0; k < numOfKernels; k++)
            totalKer[k] += 2.0 * ( labels[i] - kernelSum ) * -ker[k];
    }

    for (int k = 0; k < numOfKernels; k++)
        kernelCoefficients[k] += -current_lr * totalKer[k] - current_lr * this->beta * kernelCoefficients[k];

    delete[] ker;
    /* ------------------------------------------------ */


    /* ----------- Update embedding vectors ----------- */
//    double *neule;
//    double *g, *temp_g;
//    neule = new double[this->dim_size]{0};
//    g = new double[this->dim_size]{0};
//    temp_g = new double[this->dim_size]{0};
//
//
//    for (int i = 0; i < contextIds.size(); i++) {
//
//        for (int k = 0; k < numOfKernels; k++) {
//
//            this->get_schoenberg_grad(temp_g, labels[i], this->kernelParams[k + 1], contextIds[i], centerId, current_lr);
//
//            for (int d = 0; d < this->dim_size; d++)
//                g[d] += (1.0 / numOfKernels) * temp_g[d];
//
//        }
//
//        for (int d = 0; d < this->dim_size; d++)
//            neule[d] += g[d];
//
//        for (int d = 0; d < this->dim_size; d++)
//            this->emb1[contextIds[i]][d] += g[d] - current_lr * this->lambda * (this->emb1[contextIds[i]][d]);
//    }
//    for (int d = 0; d < this->dim_size; d++)
//        this->emb0[centerId][d] += -neule[d] - current_lr * this->lambda * (this->emb0[centerId][d]);
//
//
//    delete[] neule;
//    delete[] g;
//    delete[] temp_g;
    /* ------------------------------------------------ */

}



void Model::update_gauss_sch_multiple_kernel(vector <double> labels, vector <int> contextIds, int centerId, double current_lr, int numOfKernels, double *&kernelCoefficients) {

    /* ----------- Update embedding vectors ----------- */
    double *neule;
    double *g, *temp_g;
    double eta, e, *diff,  *z;
    double var;
    double *e_values;

    neule = new double[this->dim_size]{0};
    g = new double[this->dim_size]{0};
    temp_g = new double[this->dim_size]{0};
    z = new double[this->dim_size]{0};
    diff = new double[this->dim_size]{0};
    e_values = new double[this->dim_size]{0};
    double e_values_sum=0;
    double e_values_sum_ext=0;
    double f;

    for(int i = 0; i < contextIds.size(); i++) {

        /*
                 eta = 0.0;
        for (int d = 0; d < this->dim_size; d++)
            eta += diff[d]*diff[d];

        e_values_sum = 0;
        e_values_sum_ext = 0;
        for(int k=0; k < numOfKernels; k++) {
            var = this->kernelParams[k+1] * this->kernelParams[k+1];
            e_values[k] = kernelCoefficients[k] * exp( -eta/(2.0*var) );
            e_values_sum += e_values[k];
            e_values_sum_ext += e_values[k] * ( 1.0 / var );
        }

        f = 2.0 * ( labels[i]-e_values_sum  ) * ( e_values_sum_ext );

         */

        for (int d = 0; d < this->dim_size; d++)
            diff[d] = this->emb1[contextIds[i]][d] - this->emb0[centerId][d];

        eta = 0.0;
        for (int d = 0; d < this->dim_size; d++)
            eta += diff[d]*diff[d];
        eta = 1.0 / ( 1.0 + eta ); // eta = (1 + (x-y)^2 ) ^ {-1}

        e_values_sum = 0;
        e_values_sum_ext = 0;
        for(int k=0; k < numOfKernels/2; k++) {
            var = this->kernelParams[k+1] * this->kernelParams[k+1];
            e_values[k] = kernelCoefficients[k] * exp( -eta/(2.0*var) );
            e_values_sum += e_values[k];
            e_values_sum_ext += e_values[k] * ( 1.0 / var );
        }
        for(int k=numOfKernels/2; k < numOfKernels; k++) {
            e = pow(eta, this->kernelParams[k+1]); // ( 1 + (x-y) )^{-\alpha}
            e_values[k] = kernelCoefficients[k] * e;
            e_values_sum += e_values[k];
            e_values_sum_ext += e_values[k] * eta * ( 2.0 );
        }

        f = 2.0 * ( labels[i] - e_values_sum  ) * ( e_values_sum_ext );
        for (int d = 0; d < this->dim_size; d++)
            z[d] = f * ( diff[d] );

        for (int d = 0; d < this->dim_size; d++)
            g[d] = -current_lr * z[d]; // minus comes from the objective function, minimization

        for (int d = 0; d < this->dim_size; d++)
            neule[d] += g[d];

        for (int d = 0; d < this->dim_size; d++)
            this->emb1[contextIds[i]][d] += g[d] - current_lr*this->lambda*(this->emb1[contextIds[i]][d]);
    }
    for (int d = 0; d < this->dim_size; d++)
        this->emb0[centerId][d] += -neule[d] - current_lr*this->lambda*(this->emb0[centerId][d]);


    delete[] neule;
    delete [] g;
    delete [] temp_g;
    delete [] diff;
    delete [] z;
    delete [] e_values;
    /* ------------------------------------------------ */

    /* ---------- Update kernel coefficients ---------- */
    double kernelSum;
    double *ker, *totalKer;

    ker = new double[numOfKernels]{0};
    totalKer = new double[numOfKernels]{0};

    for(int i = 0; i < contextIds.size(); i++) {
        kernelSum = 0;
        for (int k = 0; k < numOfKernels/2; k++) {
            ker[k] = this->gaussian_kernel(contextIds[i], centerId, this->kernelParams[k + 1]);
            kernelSum += kernelCoefficients[k] * ker[k];
        }
        for (int k = numOfKernels/2; k < numOfKernels; k++) {
            ker[k] = this->schoenberg_kernel(contextIds[i], centerId, this->kernelParams[k + 1]);
            kernelSum += kernelCoefficients[k] * ker[k];
        }
        for (int k = 0; k < numOfKernels; k++)
            totalKer[k] += 2.0 * ( labels[i] - kernelSum ) * -ker[k];
    }

    for (int k = 0; k < numOfKernels; k++)
        kernelCoefficients[k] += -current_lr * totalKer[k] - current_lr * this->beta * kernelCoefficients[k];

    delete[] ker;
    /* ------------------------------------------------ */


}



void Model::get_schoenberg_grad(double *&g, double label, double alpha, int contextId, int centerId, double current_lr) {

    double eta, e, scalar_val;
    auto *diff = new double[this->dim_size]{0};


    for (int d = 0; d < this->dim_size; d++)
        diff[d] = this->emb1[contextId][d] - this->emb0[centerId][d]; // (x-y)

    eta = 0.0;
    for (int d = 0; d < this->dim_size; d++)
        eta += diff[d]*diff[d];
    eta = 1.0 + eta;
    eta = 1.0 / eta; // eta = (1 + (x-y)^2 ) ^ {-1}

    e = pow(eta, alpha); // ( 1 + (x-y) )^{-\alpha}
    scalar_val = 2.0 * ( label - e  ) * ( alpha * e * eta ) * ( 2.0 );
    for (int d = 0; d < this->dim_size; d++)
        g[d] =  scalar_val * ( diff[d] );

    for (int d = 0; d < this->dim_size; d++)
        g[d] = -current_lr * g[d]; // minus comes from the objective function, minimization


    delete [] diff;
}

void Model::inf_poly_kernel(double alpha, vector <double> labels, int centerId, vector <int> contextIds) {
    /*
    double *neule;
    double *z, *g, eta, *diff;
    double alpha_p = optionalParams[0];
    double beta_p = optionalParams[1];
    double temp1, temp2;

    neule = new double[dim_size];
    diff = new double[dim_size];
    z = new double[dim_size];
    g = new double[dim_size];

    for (int d = 0; d < dim_size; d++) {
        neule[d] = 0.0;
        diff[d] = 0.0;
    }


    for(int i = 0; i < contextIds.size(); i++) {

        for (int d = 0; d < dim_size; d++)
            diff[d] = emb1[contextIds[i]][d] - emb0[centerId][d];

        eta = 0.0;
        for (int d = 0; d < dim_size; d++)
            eta += pow(diff[d], 2.0);

        if(labels[i] > 0) { // beta^{-alpha}
            for (int d = 0; d < dim_size; d++)
                z[d] = -alpha_p * ( 2.0*diff[d] / (beta_p + eta) );
        } else {
            temp1 = (beta_p + eta);
            temp2 = alpha_p * pow(temp1, -alpha_p-1.0) / ( pow(beta_p, -alpha_p) - pow(temp1, -alpha_p) );
            for (int d = 0; d < dim_size; d++)
                z[d] = 2.0*diff[d] * temp2;
        }

        for (int d = 0; d < dim_size; d++)
            g[d] = alpha * z[d];

        for (int d = 0; d < dim_size; d++) {
            neule[d] += -g[d];
        }

        for (int d = 0; d < dim_size; d++)
            emb1[contextIds[i]][d] += g[d];
    }
    for (int d = 0; d < dim_size; d++)
        emb0[centerId][d] += neule[d];


    delete[] neule;
    delete [] diff;
    delete [] z;
    delete [] g;
     */
}

void Model::update_gaussian_multiple_kernel2(vector <double> labels, int centerId, vector <int> contextIds, double current_lr, int numOfKernels, double *kernelCoefficients) {

    double *neule, *g, *z;
    double ksum;


    /* ----------- Update embedding vectors ----------- */
    neule = new double[this->dim_size]{0};
    g = new double[this->dim_size]{0};



    for (int i = 0; i < contextIds.size(); i++) {

        ksum = 0;
        auto *e = new double[numOfKernels];
        for (int k = 0; k < numOfKernels; k++) {
            e[k] = this->gaussian_kernel(contextIds[i], centerId, this->kernelParams[k + 1]);
            ksum += e[k] / numOfKernels;
        }

        z = new double[this->dim_size]{0};
        for (int k = 0; k < numOfKernels; k++) {

            auto *temp_g = new double[this->dim_size]{0};
            get_gaussian_kernel_grad(temp_g, e[k], contextIds[i], centerId, this->kernelParams[k + 1]);
            for(int d=0; d < this->dim_size; d++)
                z[d] += temp_g[d] / numOfKernels;
            delete [] temp_g;

        }
        delete [] e;


        for(int d=0; d < this->dim_size; d++)
            g[d] = 2.0 * ( labels[i] - ksum  ) * ( -z[d] );
        delete [] z;

        for (int d = 0; d < this->dim_size; d++)
            neule[d] += g[d];

        for (int d = 0; d < this->dim_size; d++)
            this->emb1[contextIds[i]][d] += g[d] - current_lr * this->lambda * (this->emb1[contextIds[i]][d]);
    }
    for (int d = 0; d < this->dim_size; d++)
        this->emb0[centerId][d] += -neule[d] - current_lr * this->lambda * (this->emb0[centerId][d]);


    delete[] neule;
    delete[] g;
    /* ------------------------------------------------ */

}

void Model::get_gaussian_kernel_grad(double *&g, double e, int contextId, int centerId, double sigma) {

    //double e = this->gaussian_kernel(contextId, centerId, sigma);
    for (int d = 0; d < this->dim_size; d++)
        g[d] = e * ( this->emb1[contextId][d] - this->emb0[centerId][d] ) / (sigma*sigma);

}




void Model::run() {

    //default_random_engine generator;
    normal_distribution<double> normal_distr(0.0, 1.0);

    /* */
    // Initialize parameters
    uniform_real_distribution<double> real_distr(-0.5 /dim_size , 0.5/dim_size);

    for(int node=0; node<vocab_size; node++) {
        for(int d=0; d<dim_size; d++) {
            emb0[node][d] = real_distr(generator);
            emb1[node][d] = 0.0;
        }
    }

    // Get the number of kernels
    int numOfKernels = (int) this->kernelParams[0];
    // Set the kernel coefficients
    auto *kernelCoefficients = new double[numOfKernels]{0};
    double kernelCoeffSum = 0;
    for(int k=0; k<numOfKernels; k++) {
        kernelCoefficients[k] = real_distr(generator);
        //kernelCoeffSum += kernelCoefficients[k];
    }
    for(int k=0; k<numOfKernels; k++) {
        //kernelCoefficients[k] = kernelCoefficients[k] / kernelCoeffSum;
        cout << kernelCoefficients[k] << endl;
    }

    fstream fs(this->corpusFile, fstream::in);
    if(fs.is_open()) {

        string line, token, center_node, context_node;
        vector <string> nodesInLine;
        int context_start_pos, context_end_pos;
        vector <double> x;
        vector <int> contextIds;
        int centerId;
        double z, g, *neule;
        int *neg_sample_ids;
        double current_alpha = this->lr;
        int processed_node_count = 0;


        cout << "--> The update of the model parameters has started." << endl;

        for(int iter=0; iter<num_of_iters; iter++) {

            fs.clear();
            fs.seekg(0, ios::beg);
            cout << "    + Iteration: " << iter+1 << "/" << this->num_of_iters << endl;

            while (getline(fs, line)) {
                stringstream ss(line);
                while (getline(ss, token, ' '))
                    nodesInLine.push_back(token);

                for (int center_pos = 0; center_pos < nodesInLine.size(); center_pos++) {

                    // Update alpha
                    if (processed_node_count % 10000 == 0) {
                        current_alpha = this->lr * (1.0 - this->decay_rate * ((float) processed_node_count / (total_nodes*num_of_iters)));

                        if (current_alpha < this->min_lr)
                            current_alpha = this->min_lr;

                        cout << "\r    + Current alpha: " << setprecision(4) << current_alpha;
                        cout << " and " << processed_node_count-(total_nodes*iter) << "" << setprecision(3) << "("
                             << 100 * (float) ( processed_node_count-(total_nodes*iter) ) / total_nodes << "%) "
                             << "nodes in the file have been processed.";
                        cout << flush;
                    }


                    context_start_pos = max(0, center_pos - (int)this->window_size);
                    context_end_pos = min(center_pos + (int)this->window_size, (int) nodesInLine.size() - 1);

                    center_node = nodesInLine[center_pos];
                    centerId = node2Id[center_node];

                    // Resize
                    contextIds.resize((int) negative_sample_size + 1);
                    x.resize((int) negative_sample_size + 1);
                    neg_sample_ids = new int[negative_sample_size];

                    for (int context_pos = context_start_pos; context_pos <= context_end_pos; context_pos++) {

                        if (center_pos != context_pos) {
                            context_node = nodesInLine[context_pos];

                            contextIds[0] = node2Id[context_node];
                            uni.sample(negative_sample_size, neg_sample_ids);
                            for (int i = 0; i < negative_sample_size; i++)
                                contextIds[i + 1] = neg_sample_ids[i];
                            x[0] = 1.0;
                            fill(x.begin() + 1, x.end(), 0.0);

                            if(this->kernel == "nokernel") {

                                update_rule_nokernel(x, centerId, contextIds, current_alpha);

                            } else if(this->kernel == "gaussian" || this->kernel == "gauss") {

                                update_rule_gaussian_kernel(x, centerId, contextIds, current_alpha);

                            } else if(this->kernel == "schoenberg" || this->kernel == "sch") {

                                update_rule_schoenberg_kernel(x, centerId, contextIds, current_alpha);

                            } else if(this->kernel == "multi-gauss" || this->kernel == "multiple-gauss" || this->kernel == "multiple-gaussian") {

                                update_gaussian_multiple_kernel(x, centerId, contextIds, current_alpha, numOfKernels, kernelCoefficients);

                            } else if(this->kernel == "multi-sch" || this->kernel == "multiple-sch" || this->kernel == "multiple-schoenberg") {

                                update_schoenberg_multiple_kernel(x, contextIds, centerId, current_alpha, numOfKernels, kernelCoefficients);

                            } else if(this->kernel == "multi-gauss-sch" || this->kernel == "multiple-gauss-sch") {

                                update_gauss_sch_multiple_kernel(x, contextIds, centerId, current_alpha, numOfKernels, kernelCoefficients);

                            } else if(this->kernel == "multi-gauss2" || this->kernel == "multiple-gauss2" || this->kernel == "multiple-gaussian2") {

                                update_gaussian_multiple_kernel2(x, centerId, contextIds, current_alpha, numOfKernels, kernelCoefficients);

                            } else if(this->kernel == "inf_poly") {

                                //x[0] = pow(optionalParams[1], -optionalParams[0]);
                                //inf_poly_kernel(alpha, x, centerId, contextIds);

                            } else if(this->kernel == "deneme") {
                                //cout << "method2" << endl;
                                /* */

                            } else if(this->kernel == "multiple") {
                                //cout << "method2" << endl;
                                double var1 = 1.0;
                                double var2 = 2.0;
                                double var3 = 3.0;
                                double c1= 3.0;
                                double c2 = 2.0;
                                double c3 = 1.0;
                                //gaussian_kernel(alpha, x, centerId, contextIds, var1, c1);
                                //gaussian_kernel(alpha, x, centerId, contextIds, var2, c2);
                                //gaussian_kernel(alpha, x, centerId, contextIds, var3, c3);
                            } else {
                                cout << "Not a valid method name" << endl;
                            }

                        }

                    }

                    // Increase the node count
                    processed_node_count++;

                    // Clear the vectors
                    contextIds.clear();
                    x.clear();
                    delete[] neg_sample_ids;
                }


                nodesInLine.clear();

            }
            cout << endl;

        }
        fs.close();

        cout << endl << "Done" << endl;

    } else {
        cout << "An error occurred during reading file!" << endl;
    }

    for(int k=0; k<numOfKernels; k++) {
        cout << kernelCoefficients[k] << " " << endl;
    }
    cout << endl;
    delete [] kernelCoefficients;

}


void Model::save_embeddings(string file_path) {

    this->save_embeddings(file_path, 0);

}

void Model::save_embeddings(string file_path, int layerId) {

    fstream fs(file_path, fstream::out);
    if(fs.is_open()) {
        fs << vocab_size << " " << dim_size << endl;
        for(int node=0; node<vocab_size; node++) {
            fs << vocab_items[node].node << " ";
            for(int d=0; d<dim_size; d++) {
                if(layerId == 1)
                    fs << emb1[node][d] << " ";
                else
                    fs << emb0[node][d] << " ";
            }
            fs << endl;
        }

        fs.close();

    } else {
        cout << "An error occurred during opening the file!" << endl;
    }

}
