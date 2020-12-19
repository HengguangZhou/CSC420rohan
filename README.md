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
- matplotlib

### Test
To test on single image with scale factor 2 using pretrained weights:
```
python test.py --image 'image/butterfly_GT.bmp' --sr_weights 'weights/VDSR_sr.pth' --sr_module 'VDSR' --lr_weights 'weights/DVDSR_lr.pth' --lr_module 'DVDSR'
                --scale 2
```
The program will generate bicubic, downscale, and upscale images in images/
### Train
To retrain the model:
```
python train.py --data_type 'h5' --train_data 'data/91-image_x2.h5' --eval_data 'data/Set5_x2.h5' --sr_module 'VDSR' --lr_module 'DVDSR' --scale 2
```
The program will generate runs/ for tensorboard, and weights file in weights/
### Run ZSSR
To train and test on an input image:
```
python ./ZSSR/main.py --input-img './image/butterfly_GT.bmp' 
```
The program will output result.png.
### Dataset
H5 Data image file are in the data/.
If you prefer to use image folder, please organize the data repository in a similar hierarchy as below:
```
data
  ---nameofdataset
      ---train
         ---HR
         ---LR
      ---eval
         ---HR
         ---LR
```
