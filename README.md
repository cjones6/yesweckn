YesWeCKN
====================================

This code provides an implementation of convolutional kernel networks trained using the method developed in the following [paper](https://arxiv.org/pdf/1903.08131.pdf):

C. Jones, V. Roulet and Z. Harchaoui. Kernel-based Translations of Convolutional Networks. arXiv preprint arXiv:1903.08131, 2019.

If you use this code please cite the paper using the bibtex reference below.

```
@article{JRH2019,
  title={Kernel-based Translations of Convolutional Networks},
  author={Jones, Corinne and Roulet, Vincent and Harchaoui, Zaid},
  journal={arXiv preprint arXiv:1903.08131},
  year={2019}
}
```

For a more detailed description of the code and the setup instructions see the [documentation](http://www.stat.washington.edu/people/cjones6/software/yesweckn).


Introduction
-----------------
Convolutional kernel networks, first introduced by Mairal et al. (2014) and further developed by Mairal (2016) and Paulin et al. (2017), allow one to learn feature representations for images or signals in an unsupervised or in a supervised manner. In our [paper](https://arxiv.org/pdf/1903.08131.pdf) we describe a systematic way to translate a ConvNet into a CKN. Moreover, we develop an end-to-end training algorithm for CKNs and demonstrate that CKNs can achieve comparable performance to their ConvNet counterparts.

This code implements CKNs for images and other data observed on a grid and trains them using a stochastic gradient optimization method with an accurate gradient. The scripts in the experiments folder train the CKN counterparts to LeNet-1 and LeNet-5 on MNIST (LeCun et al., 1998) and All-CNN-C on CIFAR-10 (Springenberg et al., 2015; Krizhevsky and Hinton, 2009). Each architecture is specified in the cfg folder. 

Installation
-----------------
This code is compatible with Python 2.7, 3.6, and 3.7 and was written with PyTorch version 1.0.0. 

The primary dependencies are:

* PyTorch version 1.0.0 and torchvision https://pytorch.org/
* Faiss version 1.5.0 https://github.com/facebookresearch/faiss
* Scipy version 1.2.0 https://www.scipy.org/
* Matplotlib version 2.2.3 https://matplotlib.org/

The remainder of the dependencies are standard and e.g., come pre-installed with Anaconda.

The code runs on a CPU or GPU, although it is intended to be run on a GPU. To run it on a CPU change the line 
`device = torch.device('cuda:0')`
 in the file src/default_params.py to `device = torch.device('cpu')`
 
This version of the code is not compatible with older versions of PyTorch. The code has not been tested on Windows operating systems.

Contact
-----------------
You can report issues and ask questions in the repository's issues page. If you choose to send an email instead, please direct it to Corinne Jones at cjones6@uw.edu and include [yesweckn] in the subject line.

Authors
-----------------
[Corinne Jones](https://www.stat.washington.edu/people/cjones6/)  
[Vincent Roulet](http://faculty.washington.edu/vroulet/)  
[Zaid Harchaoui](http://faculty.washington.edu/zaid/)  


License
-----------------
This code has a GPLv3 license.


Acknowledgements
--------------------------
This work was supported by NSF TRIPODS Award CCF-1740551, the program "Learning in Machines and Brains" of CIFAR, and faculty research awards.


References
-----------------
- Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied to document recognition. In *Intelligent Signal Processing*, pages 306–351. IEEE Press, 2001.
- A. Krizhevsky and G. E. Hinton. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009.
- J. Mairal, P. Koniusz, Z. Harchaoui, and C. Schmid. Convolutional kernel networks. In *Advances in Neural Information Processing Systems*, pages 2627–2635, 2014.
- J. Mairal. End-to-end kernel learning with supervised convolutional kernel networks. In *Advances in Neural Information Processing Systems*, pages 1399–1407, 2016.
- A. Paszke, S. Gross, S. Chintala, G. Chanan, E. Yang, Z. DeVito, Z. Lin, A. Desmaison, L. Antiga, and A. Lerer. Automatic differentiation in PyTorch. *NIPS 2017 Workshop on Autodiff*, 2017.
- M. Paulin, J. Mairal, M. Douze, Z. Harchaoui, F. Perronnin, and C. Schmid. Convolutional patch representations for image retrieval: An unsupervised approach. *International Journal of Computer Vision*, 121(1):149–168, 2017.
- J. Springenberg, A. Dosovitskiy, T. Brox, and M. Riedmiller. Striving for simplicity: The all convolutional net. In *arXiv:1412.6806, also appeared at ICLR 2015 Workshop Track*, 2015.
