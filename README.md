# P2-ML-GCN and RP2-ML-GCN.pytorch

### Requirements
Please, install the following packages
- numpy
- torch-0.3.1
- torchnet
- torchvision-0.2.0
- tqdm

Please unzip the files data.zip and checkpoint.zip.

### Download pretrain models
checkpoint/coco ([GoogleDrive](https://drive.google.com/open?id=1ivLi1Rc-dCUmN1ProcMk76zxF1DSvlIk))

checkpoint/voc ([GoogleDrive](https://drive.google.com/open?id=1lhbmW5g-Mo9KgI07nmc1kwSbEnb6t-YA))

### Laplace Noise
```sh
noise = torch.from_numpy(np.random.laplace(0,1/epsilon,feature.size())).cuda().float()
```
You can add the noise Laplace noise into the model's input features, weights and output (in P2_models.py).

### Regularization term 
The regularization term can be added into the original loss function (in RP2_demo_voc2007_gcn.py or RP2_demo_voc2007_gcn.py).


### Train Voc2007
```sh
python3 RP2_demo_voc2007_gcn.py data/voc --image-size 448 --batch-size 16 --epochs 40
```
### Train MS-COCO
```sh
python3 RP2_demo_coco_gcn.py data/coco --image-size 448 --batch-size 8 --epochs 20
```

### Test Voc2007
```sh
python3 RP2_demo_voc2007_gcn.py data/voc --image-size 448 --batch-size 16 -e --resume checkpoint/voc/voc_checkpoint.pth.tar
```

### Test MS-COCO
```sh
python3 RP2_demo_coco_gcn.py data/coco --image-size 448 --batch-size 8 -e --resume checkpoint/coco/coco_checkpoint.pth.tar
```

### Training Framework
The framework follows ML-GCN [Multi-Label Image Recognition with Graph Convolutional Networks](https://arxiv.org/abs/1904.03582), CVPR 2019.
And add the Laplace noise into the dot-produce of features and weights.
![Framework](https://github.com/ahahnut/-R-P2-ML-GCN/blob/master/Framework.png)

## Reference
This project is based on https://github.com/Megvii-Nanjing/ML-GCN

## Tips
If you have any questions about our work, please do not hesitate to contact us.

