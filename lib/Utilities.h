#ifndef UTILITIES_H
#define UTILITIES_H
#include <string>
#include <sstream>
#include <vector>
#include <iostream>


namespace Constants
{
    const std::string ProgramName = "kernelNE";
};

using namespace std;



int parse_arguments(int argc, char** argv, string &corpusFile, string &embFile, string &kernel, double *&kernelParams,
                    unsigned int &dimension, unsigned int &window, unsigned int &neg,
                    double &lr, double &min_lr, double &decay_rate, double &lambda, double &beta, unsigned int &iter,
                    bool &verbose) {

    vector <string> parameter_names{"--help",
                                    "--corpus", "--emb", "--kernel", "--params"
,                                    "--dim", "--window", "--neg",
                                    "--lr", "--min_lr", "--decay_rate", "--lambda", "--beta", "--iter",
                                    "--verbose"
    };

    string arg_name;
    stringstream help_msg, help_msg_required, help_msg_opt;

    // Set the help message
    help_msg_required << "\nUsage: ./" << Constants::ProgramName;
    help_msg_required << " " << parameter_names[1] << " CORPUS_FILE "
                      << parameter_names[2] << " EMB_FILE "
                      << parameter_names[3] << " KERNEL " << endl;

    help_msg_opt << "\nOptional parameters:\n";
    help_msg_opt << "\t[ " << parameter_names[4] << " (Default: " << kernelParams[0] << " " << kernelParams[1] << ") ]\n";
    help_msg_opt << "\t[ " << parameter_names[5] << " (Default: " << dimension << ") ]\n";
    help_msg_opt << "\t[ " << parameter_names[6] << " (Default: " << window << ") ]\n";
    help_msg_opt << "\t[ " << parameter_names[7] << " (Default: " << neg << ") ]\n";
    help_msg_opt << "\t[ " << parameter_names[8] << " (Default: " << lr << ") ]\n";
    help_msg_opt << "\t[ " << parameter_names[9] << " (Default: " << min_lr << ") ]\n";
    help_msg_opt << "\t[ " << parameter_names[10] << " (Default: " << decay_rate << ") ]\n";
    help_msg_opt << "\t[ " << parameter_names[11] << " (Default: " << lambda << ") ]\n";
    help_msg_opt << "\t[ " << parameter_names[12] << " (Default: " << beta << ") ]\n";
    help_msg_opt << "\t[ " << parameter_names[13] << " (Default: " << iter << ") ]\n";
    help_msg_opt << "\t[ " << parameter_names[14] << " (Default: " << verbose << ") ]\n";
    help_msg_opt << "\t[ " << parameter_names[0] << ", -h ] Shows this message";

    help_msg << "" << help_msg_required.str() << help_msg_opt.str();

    // Read the argument values
    for(int i=1; i<argc;) {

        arg_name.assign(argv[i]);

        if (arg_name.compare(parameter_names[1]) == 0) {
            corpusFile = argv[++i];
        } else if (arg_name.compare(parameter_names[2]) == 0) {
            embFile = argv[++i];
        } else if (arg_name.compare(parameter_names[3]) == 0) {
            kernel = argv[++i];
        } else if (arg_name.compare(parameter_names[4]) == 0) {

            int numOfValues = stoi(argv[++i]);
            if(numOfValues > 1) {
                delete [] kernelParams;
                kernelParams = new double[numOfValues+1];
                kernelParams[0] = (double) numOfValues;
            }
            for(int k=1; k <= numOfValues; k++)
                kernelParams[k] = stod(argv[++i]);

        } else if (arg_name.compare(parameter_names[5]) == 0) {
            dimension = stoi(argv[++i]);
        } else if (arg_name.compare(parameter_names[6]) == 0) {
            window = stoi(argv[++i]);
        } else if (arg_name.compare(parameter_names[7]) == 0) {
            neg = stoi(argv[++i]);
        } else if (arg_name.compare(parameter_names[8]) == 0) {
            lr = stod(argv[++i]);
        } else if (arg_name.compare(parameter_names[9]) == 0) {
            min_lr = stod(argv[++i]);
        } else if (arg_name.compare(parameter_names[10]) == 0) {
            decay_rate = stod(argv[++i]);
        } else if (arg_name.compare(parameter_names[11]) == 0) {
            lambda = stod(argv[++i]);
        } else if (arg_name.compare(parameter_names[12]) == 0) {
            beta = stod(argv[++i]);
        } else if (arg_name.compare(parameter_names[13]) == 0) {
            iter = stoi(argv[++i]);
        } else if (arg_name.compare(parameter_names[14]) == 0) {
            verbose = stoi(argv[++i]);
        } else if (arg_name.compare(parameter_names[0]) == 0 or arg_name.compare("-h") == 0) {
            cout << help_msg.str() << endl;
            return 1;
        } else {
            cout << "Invalid argument name: " << arg_name << endl;
            return -2;
        }
        arg_name.clear();

        i++;

    }

    // Print all the parameter settings if verbose is set
    if(verbose) {
        cout << "--> Parameter settings." << endl;
        cout << "\t+ Kernel: " << kernel << endl;
        cout << "\t+ KernelParameters:";
        for(int k=1; k<=kernelParams[0]; k++)
            cout << kernelParams[k] << " ";
        cout << endl;
        cout << "\t+ Dimension: " << dimension << endl;
        cout << "\t+ Window size: " << window << endl;
        cout << "\t+ Negative samples: " << neg << endl;
        cout << "\t+ Starting learning rate: " << lr << endl;
        cout << "\t+ Minimum learning rate: " << min_lr << endl;
        cout << "\t+ Decay rate: " << decay_rate << endl;
        cout << "\t+ Lambda: " << lambda << endl;
        cout << "\t+ Beta: " << beta << endl;
        cout << "\t+ Number of iterations: " << iter << endl;
    }

    // Check if the required parameters were set or not
    if(corpusFile.empty() || embFile.empty() || kernel.empty() ) {
        cout << "Please enter the required parameters: ";
        cout << help_msg_required.str() << endl;

        return -4;
    }

    // Check if the constraints are satisfied
    if( dimension < 0 ) {
        cout << "Dimension size must be greater than 0!" << endl;
        return -5;
    }
    if( window < 0 ) {
        cout << "Window size must be greater than 0!" << endl;
        return -5;
    }
    if( neg < 0 ) {
        cout << "The number of negative samples must be greater than 0!" << endl;
        return -5;
    }
    if( kernel != "nokernel" &&
        (kernel != "gauss" && kernel != "gaussian") &&
        (kernel != "sch" && kernel != "schoenberg" ) &&
        (kernel != "multi-gauss" && kernel == "multiple-gauss" && kernel == "multiple-gaussian") &&
        (kernel != "multi-sch" && kernel == "multiple-sch" && kernel == "multiple-schoenberg") &&
        (kernel != "multi-gauss2" && kernel == "multiple-gauss2" && kernel == "multiple-gaussian2") &&
        (kernel != "multi-gauss-sch" && kernel == "multiple-gauss-sch") ) {
            cout << "The kernel name must be gaussian, schoenberg or multiple-gaussian! " << kernel << endl;
        return -6;
    }

    return 0;

}

#endif //UTILITIES_H