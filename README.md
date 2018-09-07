# Implementations of deep learning networks for computer vision in tensorflow

Tensorflow (tf) implementations of: 

* [U-Net](https://github.com/josegcpa/tf_implementations/blob/master/u-net.py) [[1](https://arxiv.org/abs/1505.04597)] 
* [A variational auto-encoder](https://github.com/josegcpa/tf_implementations/blob/master/vae.py) (VAE) [[2](https://arxiv.org/abs/1312.6114)]
  * I would also recommend this for a clearer understanding [[3](https://arxiv.org/abs/1606.05908)]
* [A region proposal network](https://github.com/josegcpa/tf_implementations/blob/master/rpn_with_anchors.py) (RPN) [[4](https://arxiv.org/abs/1504.08083)], featuring:
  * Anchors [[5](https://arxiv.org/abs/1506.01497)]
  * A feature pyramid network (FPN) [[6](https://arxiv.org/abs/1612.03144)]
    * MobileNetV2 backbone [[7](https://arxiv.org/abs/1801.04381)]
    
This is not extremely organised for now;
In the future input examples will be added to this repository to facilitate usage.

These have all been implemented and tested with tensorflow 1.6 for GPU installed from source with CUDA 8.0 and CuDNN 6.0 and a Quadro M6000 GPU.
