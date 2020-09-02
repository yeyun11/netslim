# Channel Pruning: Network Slimming

This repo implements the following paper in [PyTorch](https://pytorch.org):  

**Network Slimming**. By training networks with L1 penalty, and then prune channels with smaller scaling factors of BN. Details are described in:

[**Learning Efficient Convolutional Networks Through Network Slimming**](http://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.pdf)

## No Dirty Work

Different with most other popular network slimming repos, this implementation enables training & pruning networks with a few lines of new codes. Writing codes specifically for pruning is not required. 

BN layers are automatically identified by 1) parse the traced graph by [TorchScript](https://pytorch.org/docs/stable/jit.html), and 2) identify **prunable BN layers** which only have Convolution(groups=1)/Linear in preceding & succeeding layers. Example of a **prunable BN**:

            conv/linear --> ... --> BN --> ... --> conv/linear
                                     |
                                    ...
                                     | --> relu --> ... --> conv/linear
                                             |
                                             | --> ... --> maxpool --> ... --> conv/linear
The pruned model can be further accelerated and deployed with toolkits supporting ONNX, enabling deployment with other popular toolkits, such as [**TensorFlow**](https://www.tensorflow.org/)(with [**onnx-tensorflow**](https://github.com/onnx/onnx-tensorflow)) and [**OpenVINO**](https://software.intel.com/en-us/openvino-toolkit). An example of further acceleration with OpenVINO is included in this repo.

### Skip Connection

This repo is able to handle networks with skip-connections, but some tricks are not fully supported yet. For example [channel_selection](https://github.com/Eric-mingjie/network-slimming/blob/master/models/channel_selection.py) is supported and studied in our tech report, but the performance is not fully tested. Moreover for easier implementation, we disable channels after BN, which is actually not the efficient way. For practical cases, you may turn channel selection off or re-implement an efficient version by disabling channels before BN. Note that if it is turned off, the prunable BNs for some network architectures (ResNet/DenseNet/...) will be less than the [official implementation](https://github.com/Eric-mingjie/network-slimming). Similarly we reported the FLOPs by removing the branch in a ResBlock, but didn't really prune the branch in this code. Instead of removing a branch, we set all the weights to zeros and disable bias. For more details please refer to source code and the tech report. 

It is supposed to support user defined models with Convolution(groups=1)/Linear and BN layers. The code is tested with the [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html) examples in this repo, and an in-house [Conv3d](https://pytorch.org/docs/stable/nn.html#conv3d) based model for video classification. 

<font size=2> \* ***DataParalell*** is not supported </font>

## Known Issues

### Node without an explicit name in traced graph
This code depends on traced graph by [TorchScript](https://pytorch.org/docs/stable/jit.html), so any graph without an explicit module name will fail. For example:

   ```python
   def foward(self, x)
       # ...
       for name, child in self.backbone.named_children():
           x = child(x)
       # ...
   ```
### shortcut from BN to BN

...

### Does not support PyTorch >= 1.5

...

## Requirements

Python >= 3.6  
1.4.0 >= torch >= 1.1.0  
torchvision >= 0.3.0  
thop >= 0.0.31  

## Usage

1. Import from [netslim](./netslim) in your training script
   
     ```python
   from netslim import update_bn
   ```
   
2. Insert *updat_bn* between *loss.backward()* and *optimizer.step()*. The following is an example:

   *before*

   ```python
   # ...
   loss.backward()
   optimizer.step()
   # ...
   ```

   *after*

      ```python
   # ...
   loss.backward()
   update_bn(model)  # or update_bn(model, s), specify s to control sparsity on BN
   optimizer.step()
   # ...
      ```

   <font size=2> \* ***update_bn*** puts L1 regularization on all BNs. Sparsity on prunable BNs only is also supported for networks with complex connections, such as ResNet. Your may also specify BNs by using ***update_bn_by_names***. Check the source code for more details. </font>

3. Prune the model after training

   ```python
   from netslim import prune, network_slimming # you may import other available methods
   # For example, input_shape for CIFAR is (3, 32, 32)
   pruned_model = prune(model, input_shape) # by default, use network slimming
   
   # The following code is an example using network slimming with a prune ratio of 0.4
   #pruned_model = prune(model, input_shape, prune_method=network_slimming, prune_ratio=0.4)
   ```

4. Fine-tune & export model

5. Load the pruned model and have fun

   ```python
   from netslim import load_pruned_model
   model = MyModel()
   weights = torch.load("/path/to/pruned-weights.pth")
   pruned_model = load_pruned_model(model, weights)
   # ...
   ```

## CIFAR-100 examples

   ```shell
sh examples.sh
   ```

Check the scripts included for more usage.

## Prune & Evaluate ResNet/DenseNet with Channel Selection and Branch Removing

Examples to prune ResNet/DenseNet with channel selection and branch removing are included in the code. Check ***train-cifar.py*** and ***train-ilsvrc12.py*** for more details. As mentioned, we set weights to zeros instead of physically removing some parts of weights in ResNet/DenseNet. Corresponding modifications are required to evaluate FLOPs. 

**Evaluating FLOPs**. For VGG or other similar architectures, we use [**thop**](https://github.com/Lyken17/pytorch-OpCounter) to evaluate FLOPs. For ResNet/DenseNet, changes are needed to calculate the correct FLOPs. As an example, we made the following changes to **thop**:

1. For *count_convNd* and *count_bn*, add a conditional check if the input are all zeros. For example:

   ```python
   def count_convNd(m, x, y):
       if m.weight.data.abs().sum().item() > 0:
           # ...
   
   def count_bn(m, x, y):
       if m.weight.data.abs().sum().item() > 0 and m.bias.data.abs().sum().item() > 0:
           # ...
   ```
   
2. For channel selection, we implemented with *MaskedBatchNorm* in [**prune.py**](https://github.com/yeyun11/netslim/blob/master/netslim/prune.py). Corresponding hook has to be defined:

   ```python
   # ...
   def count_masked_bn(m, x, y):
       nelements = m.channel_indexes.size()
       # subtract, divide, gamma, beta
       total_ops = 4 * nelements
   
       m.total_ops += torch.Tensor([int(total_ops)])
   ```

3. Register hooks using *custom_ops* argument in [*profile*](https://github.com/Lyken17/pytorch-OpCounter/blob/master/thop/profile.py).
   
4. In [***profile.py***](https://github.com/Lyken17/pytorch-OpCounter/blob/master/thop/profile.py), for *count_convNd* and *count_bn*, add conditional check if the input are all zeros. For example:

   ```python
   # ...
   def profile(model, inputs, custom_ops=None, verbose=True):
       # ...
       for m in model.modules():
           # ...
           if hasattr(m, "weight") and m.weight.data.abs().sum().item() == 0:
               continue
   # ...
   ```

## Inference using [OpenVINO](https://software.intel.com/en-us/openvino-toolkit)

The efficiency of pruned model can be further improved by using OpenVINO, if you are working with Intel processors/accelerators.  As an example:

1. Download OpenVINO from the official website [OpenVINO](https://software.intel.com/en-us/openvino-toolkit).

2. After installation, initialize the environment:

   ```shell
   source /opt/intel/openvino/bin/setupvars.sh
   ```

3. Convert the pruned model to ONNX:

   ```shell
   python convert2onnx.py /path/to/pruned_model.pth
   ```
   
   For your own model, you may modify accordingly based on this script. 
   
4. Convert to OpenVINO IR using OpenVINO model optimizer:

   ```shell
   python /path/to/intel/openvino/deployment_tools/model_optimizer/mo_onnx.py --input_model /path/to/pruned_model.onnx --input_shape [input shape] --data_type [FP16|FP32|INT8] --model_name [model name]
   ```


More details about OpenVINO model optimizer can be found at [Converting a ONNX* Model](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_ONNX.html)

## Optimal Thresholding
Inspired by Network Slimming, we further proposed an extremely simple yet effective method, termed Optimal Thresholding, to avoid the over- and under-pruning. 

It works on scaling factors too, but prune negligible channels considering the distribution of scaling factors rather than the magnitudes across the whole network. Compared to Network Slimming, it shows advantages particularly for models with very high pruned percentage. Details can be found in the tech report:

[**Channel Pruning via Optimal Thresholding**](https://arxiv.org/pdf/2003.04566.pdf)

Due to the company's policy, The author will release his implementation sometime later. If you are interested you may also follow the paper to implement based on this repo or any other network slimming code. It should only take a little effort since OT is extremely simple. 

## Acknowledgement

The implementation of ***udpate_bn*** is referred to [pytorch-slimming](https://github.com/foolwood/pytorch-slimming).
