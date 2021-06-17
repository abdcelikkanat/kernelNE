## Multiple Kernel Representation Learning on Networks

#### Description
We bring together the best of both worlds, towards learning node representations. In particular, we propose a weighted matrix factorization model that encodes random walk-based information about nodes of the network. The benefit of this novel formulation is that it enables us to utilize kernel functions without realizing the exact proximity matrix so that it enhances the expressiveness of existing matrix decomposition methods with kernels and alleviates their computational complexities. We extend the approach with a multiple kernel learning formulation that provides the flexibility of learning the kernel as the linear combination of a dictionary of kernels in data-driven fashion. We perform an empirical evaluation on real-world networks, showing that the proposed model outperforms baseline node embedding algorithms in downstream machine learning tasks.
#### Compilation
**1.** Firstly, create a folder named "build."
```
mkdir build
```

**2.** Navigate to the directory and run the "cmake" command.
```
cd build
cmake ..
```

**3.** Compile the codes by typing the following command:
```
make all
```
#### Learning Representations

**1.** You can learn the node representations by running
```
kernelNE --corpus CORPUS_FILE --emb EMB_FILE --kernel KERNEL
```
**2.** To see the detailed parameter settings, you can use
```
kernelNE --help
```

#### References
A. Celikkanat and F. D. Malliaros, [Kernel Node Embeddings](https://doi.org/10.1109/GlobalSIP45357.2019.8969363), 7th IEEE Global Conference on Signal and Information Processing (GlobalSIP), 2019 

A. Celikkanat, Y. Shen and F. D. Malliaros, [Multiple Kernel Representation Learning on Networks](https://arxiv.org/abs/2106.05057), Manuscript, 2021