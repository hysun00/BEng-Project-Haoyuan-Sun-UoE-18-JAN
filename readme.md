# Welcome to Haoyuan's BEng Project. 

# You are now at the top level of this repository.



## What's included

You have access to these four folders, namely: HLS_PRJ, LeNet-tf, overlay, SoC_PRJ.

| Folder        NAME                                           | Description                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [LeNet-tf](https://github.com/hysun00/BEng-Project-Haoyuan-Sun-UoE/tree/master/LeNet-tf) | The LeNet-5[1] model constructed with tensorflow[2] are stored in this folder. Here you can also find MNIST[3] dataset and all the parameters the LeNet-5 model needs to do the inference. For more details, please refer to the readme.md in /LeNet. |
| [HLS_PRJ](https://github.com/hysun00/BEng-Project-Haoyuan-Sun-UoE/tree/master/HLS_PRJ) | The HLS projects for generating the IP cores aiming at accelerating the CNN inference are stored here. For more details, please refer to the readme.md in /HLS_PRJ |
| [SoC_PRJ](https://github.com/hysun00/BEng-Project-Haoyuan-Sun-UoE/tree/master/SoC_PRJ) | The soc projects for configuring my PYNQ[4] board(an amazing SoC platform incorporating the productivity of python and flexibility of Xilinx SoC platform)are stored in this folder. In this folder you can find |
| [overlay](https://github.com/hysun00/BEng-Project-Haoyuan-Sun-UoE/tree/master/overlay) | The python programmes  designed for LeNet-5 inference on SoC are stored in this folder. For more details, please refer to the readme.md in /overlay |



## V1 and V2

In most of the folders you can see v1 and v2. They represents version 1 and 2, respectively; and v2 is an optimised version based on v1. Before you go further, you may want to know the difference so that unnecessary puzzlement can be avoided. You can also skip this section if you find it too early to dive into the details. The difference will be further emphasized in those four folders. 

|      | Data Precision of the LeNet  | Parallelism of the convolution IP                            | Latency of inferring a 28*28  grayscale image |
| ---- | ---------------------------- | ------------------------------------------------------------ | --------------------------------------------- |
| V1   | Floating numbers of 32 bits. | No parallelism applied.                                      | 236 ms                                        |
| v2   | Integers of 16 bits.         | Input channel parallelism applied. 8 input channels are in parallel. | 36 ms                                         |



## Remote access to PYNQ and Juypter Notebooks

By following my instructions, you can let PYNQ-Z1 board recognise hand-written digits. If you have one, then you just need to follow further instructions in [overlay](https://github.com/hysun00/BEng-Project-Haoyuan-Sun-UoE/tree/master/overlay). Otherwise, please read this section.

You can log into my PYNQ board remotely. If you want to do this, please send me an email so that I can power on the board for you. What you need to do then is to type **[The IP address](https://github.com/hysun00/BEng-Project-Haoyuan-Sun-UoE/blob/master)** (this will be delivered to you via the email) in the browser (firefox and Chrome recommended). Thereafter you will be brought to the log-in page which looks like:

![](/pic_for_readme/image-01-17-225036.png)

The password is: **xilinx**. By following the instructions you should be able to see the contents in my juypter notebook.

![](/pic_for_readme/image-20220117225426434.png)

In the folders with the suffix v1 and v2 are stored the API for IP core built with PL resources, the python programme that guides the ARM CPU to control the data flow, the input 28*28 input feature maps and parameters of LeNet-5 model, and the files configuring the ZYNQ board, which are exported from the vivado projects.



## Permission and Accessibility

After you log into the system, you are technically allowed to alter the folder or files on the PYNQ board. But please **DO NOT**  apply any change to all the files and folders without any discussion, unless otherwise suggested. 



## References

[1] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, november 1998.

[2] https://www.tensorflow.org/

[3] http://yann.lecun.com/exdb/mnist/

[4] http://www.pynq.io/community.html