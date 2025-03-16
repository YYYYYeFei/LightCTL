LightCTL: lightweight contrastive TCR-pMHC specificity learning with context-aware prompt
================================================

Installation
------------

  ```sh
  git clone https://github.com/YYYYYeFei/LightCTL.git
  cd LightCTL

  conda create --name LightCTL python=3.8.13
  conda activate LightCTL
  pip install -r requirements.txt
  ```

For training
--------
```sh
#example
cd mode
python train.py --GPU_id 0
```

For testing

Due to data size limitations, the datasets for pMTnet_train, VDJDB, McPAS-TCR, 10X, PIRD, and COVID are stored in Google Drive. You can download them from the following link:[ Google Drive](https://drive.google.com/drive/folders/1oYBUhJaDTzTAWOX4LTXv6skzY84lf-xK?usp=sharing).

--------
```sh
#example
cd mode
python test.py --GPU_id 0 --test_data_name covid
```
