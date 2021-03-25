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

Introduction
-----------------
Convolutional kernel networks, first introduced by Mairal et al. (2014) and further developed by Mairal (2016) and Paulin et al. (2017), allow one to learn feature representations for images or signals in an unsupervised or in a supervised manner. In our [paper](https://arxiv.org/pdf/1903.08131.pdf) we describe a systematic way to transform a ConvNet into a CKN. Moreover, we develop an end-to-end training algorithm for CKNs and demonstrate that CKNs can often achieve comparable performance to their ConvNet counterparts.

This code implements CKNs for images and other data observed on a grid and trains them using a stochastic gradient optimization method with an accurate gradient. The scripts in the experiments folder train the CKN counterparts to LeNet-1 and LeNet-5 on MNIST (LeCun et al., 1998), All-CNN-C on CIFAR-10 (Springenberg et al., 2015; Krizhevsky and Hinton, 2009), and AlexNet on a subset of ImageNet (Krizhevsky et al., 2012; Krizhevsky 2014). Each architecture is specified in the cfg folder. 

Installation
-----------------
This code is compatible with Python 3.7 and was written with PyTorch version 1.2.0. 

The primary dependencies are:

* Faiss version 1.5.2 https://github.com/facebookresearch/faiss
* Numpy version 1.16.4 https://numpy.org/
* PyTorch version 1.2.0 and Torchvision https://pytorch.org/
* Scipy version 1.2.1 https://www.scipy.org/

The code can run on a CPU or GPU, although it is intended to be run on a GPU. It is currently set to run on a GPU. To run it on a CPU change the line 
`device = torch.device('cuda:0')`
 in the file src/default_params.py to `device = torch.device('cpu')`
 
This version of the code is not compatible with older versions of PyTorch. The code has only been tested on Linux operating systems.


Running the code
-----------------
The scripts to reproduce the experiments are in the `experiments` folder. You will need to provide the path to the data folder as the argument `data_path`. 


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