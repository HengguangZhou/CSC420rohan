# CSC420rohan
### Dependencies
- Python
- CUDA
- Pytorch < 1.7.0
- Numpy
- PIL
- tqdm
- tensorboard
- h5py

### Test
To test on single image with scale factor 2 using pretrained weights:
```
python test.py --data_type 'h5' --image 'image/butterfly_GT.bmp' --sr_weights 'weights/VSDR_sr.pth' --sr_module 'VSDR' --lr_weights 'weights/DVSDR_lr.pth' --lr_module 'DVSDR'
                --scale 2
```

### Train
To retrain the model:
```
python train.py --data_type 'h5' --train_data 'data/91-image_x2.h5' --eval_data 'data/Set5_x2.h5' --sr_module 'VSDR' --lr_module 'DVSDR' --scale 2
```

### Dataset
Data image file are in the data/
