MDFL
======
**This is an implementation of  Hyperspectral Image Super-Resolution via Multi-domain Feature Learning.**

Motivation
=======
**When the spatial and spectral information can be extracted, the key issue is how to combine multi-domain features to learn more effective information, achieving complementation. Besides,  at present, there has been very little research by employing 2D/3D convolution to build the model. Therefore, it still needs more research efforts**

Flowchat
=====
![Image text](https://github.com/qianngli/Images/blob/master/mdfl.jpg)

**We propose a multi-domain feature learning network in alternate manner for hyperspectral image SR.  Overall, the architecture of the proposed method contains four parts, covering initial feature extraction, multi-domain feature learning module (MDFL) with 2D/3D unit, multi-domain feature fusion (MDFF), and  image reconstruction.  Specifically, a multi-domain feature  learning strategy is proposed to explore the spatial and spectral knowledge by sharing spatial information. To better fuse those feature from different domains,  the multi-domain features fusion module is introduced to learn more effective information, so as to further realize information complementation. Moreover, to recover the more edge details, we design the edge generation mechanism  to explicitly  enable the network provide priori edge.**

Dataset
------
**Two public datasets, i.e., [CAVE](https://www1.cs.columbia.edu/CAVE/databases/multispectral/ "CAVE") and [Harvard](http://vision.seas.harvard.edu/hyperspec/explore.html "Harvard"), are employed to verify the effectiveness of the  proposed MCNet. Since there are too few images in these datasets for deep learning algorithm, we augment the training data. With respect to the specific details, please see the implementation details section.**

**Moreover, we also provide the code about data pre-processing in folder [data pre-processing](https://github.com/qianngli/MCNet "data pre-processing"). The folder contains three parts, including training set augment, test set pre-processing, and band mean for all training set.**

Requirement
---------
**python 3.7, Pytorch 1.7.0, cuda 10.1**

Training
--------
**The ADAM optimizer with beta_1 = 0.9, beta _2 = 0.999 is employed to train our network.  The learning rate is initialized as 10^-4 for all layers, which decreases by a half at every 35 epochs.**

**You can train or test directly from the command line as such:**

###### # python train.py --cuda --datasetName CAVE  --upscale_factor 4
###### # python test.py --cuda --model_name checkpoint/model_4_epoch_xx.pth

 Result
--------
**To qualitatively measure the proposed method, three evaluation methods are employed to verify the effectiveness of the algorithm, including  peak signal-to-noise ratio (PSNR), structural similarity (SSIM), and spectral angle mapping (SAM).**


| Scale  |  CAVE |  Harvard |
| :------------: | :------------: | :------------: |  
|  x4 |  39.283 / 0.9328 / 3.182 | 40.177 / 0.9380 / 2.393  | 
|  x8 |  35.422 / 0.8834 / 4.385  |  35.250 / 0.8749 / 2.892 |    

**We also provide test results  for scale X4 on two datasets in terms of spatial reconstruction and spectral distortions. The results  are  [here](https://drive.google.com/drive/folders/1n6j3tv1pJ34hzu0SOp2px9qW6zl4R_uo?usp=sharing).**


--------
If you has any questions, please send e-mail to liqmges@gmail.com.


