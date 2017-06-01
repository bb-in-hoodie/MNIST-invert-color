# MNIST Invert Color

Inverting the color of MNIST images from black to white and white to black using CycleGAN implemented with PyTorch.



## Prerequites
* [Python 3.6](https://www.continuum.io/downloads)
* [PyTorch 0.1.12](http://pytorch.org/)


<br>

## Usage

#### Train the images

```bash
$ python CycleGAN_MNIST_Invert.py
```
<br>

## Results

#### 1) Black Background -> White Background
Iteration Number            |  Original Image            |  Generated Image
:-------------------------:|:-------------------------:|:-------------------------:
0            |  ![alt text](result/examples/0_BtoW_a.png)  |  ![alt text](result/examples/0_BtoW_b.png)
15            |  ![alt text](result/examples/15_BtoW_a.png)  |  ![alt text](result/examples/15_BtoW_b.png)
300            |  ![alt text](result/examples/300_BtoW_a.png)  |  ![alt text](result/examples/300_BtoW_b.png)


#### 2) White Background -> Black Background
Iteration Number            |  Original Image            |  Generated Image
:-------------------------:|:-------------------------:|:-------------------------:
0            |  ![alt text](result/examples/0_WtoB_a.png)  |  ![alt text](result/examples/0_WtoB_b.png)
15            |  ![alt text](result/examples/15_WtoB_a.png)  |  ![alt text](result/examples/15_WtoB_b.png)
300            |  ![alt text](result/examples/300_WtoB_a.png)  |  ![alt text](result/examples/300_WtoB_b.png)



