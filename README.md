# pytorch-cifar10
Training CIFAR10 with pytorch built with various models:

## Models and Accuracy
| Model             |Accuracy|
| ----------------- |--------|
| [Lenet](http://yann.lecun.com/exdb/lenet/)            |72%|
| [VGG16](https://arxiv.org/abs/1409.1556)              | 
| [ResNet18](https://arxiv.org/abs/1512.03385)          |
| [ResNet50](https://arxiv.org/abs/1512.03385)          |
| [ResNet101](https://arxiv.org/abs/1512.03385)         |
| [DenseNet121](https://arxiv.org/abs/1608.06993)       |

## Project Structure
--------------

```
├── model               - this folder contains any model of your project.
│   └── lenet.py
├──  lib  
│      ├──data_loader.py  - here's the data_generator that is responsible for all data handling.
│      ├── train.py       - this folder contains trainers of your project.
│      ├── test.py        - this folder contains trainers of your project.
│      └── utils.py
├──  main.py              - here's the main(s) of your project (you may need more than one main).
│   
```
