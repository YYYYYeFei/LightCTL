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
--------
```sh
#example
cd mode
python test.py --GPU_id 0 --test_data_name covid
```
