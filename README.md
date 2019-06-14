```
usage: train.py [-h] [--batches BATCH_SIZE] [--epochs NUM_EPOCHS]
                [--disable-cuda]
                train test

Train model

positional arguments:
  train                 path to trainining dataset
  test                  path to testing dataset

optional arguments:
  -h, --help            show this help message and exit
  --batches BATCH_SIZE  number of samples to propogate (default: 64)
  --epochs NUM_EPOCHS   number of passes through dataset (default: 32)
  --disable-cuda        disable CUDA support
```
