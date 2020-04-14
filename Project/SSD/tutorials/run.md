
## Train
Train for:
1. MNIST:
```bash
python train.py  configs/mnist.yaml
```
2. MNIST on the tdt4265.idi.ntnu.no server (This only changes the "datasets" variable in the config file):
```bash
python train.py  configs/mnist_tdt4265_server.yaml
```
3. VGG SSD300 on the PACAL VOC dataset
```bash
python train.py  configs/vgg_ssd300_voc0712.yaml
```

## Evaluate
Run test.py to evaluate the whole validation dataset: 
```bash
python test.py configs/mnist.yaml
```
Remember to give the correct config path

## Demo
For Pascal VOC
```bash
python demo.py configs/vgg_ssd300_voc0712.yaml --images_dir demo/voc --dataset_type voc
```

For MNIST Object detection
```bash
python demo.py  configs/mnist.yaml --images_dir demo/voc --dataset_type mnist
```

You will see a similar output:
```text
(0001/0005) 004101.jpg: objects 01 | load 010ms | inference 033ms | FPS 31
(0002/0005) 003123.jpg: objects 05 | load 009ms | inference 019ms | FPS 53
(0003/0005) 000342.jpg: objects 02 | load 009ms | inference 019ms | FPS 51
(0004/0005) 008591.jpg: objects 02 | load 008ms | inference 020ms | FPS 50
(0005/0005) 000542.jpg: objects 01 | load 011ms | inference 019ms | FPS 53
```