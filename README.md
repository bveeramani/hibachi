```
usage: train.py [-h] [--test TEST_DATASET] [--batches BATCH_SIZE]
                [--epochs NUM_EPOCHS] [--save filename] [--disable-cuda]
                train

Train model

positional arguments:
  train                 path to trainining dataset

optional arguments:
  -h, --help            show this help message and exit
  --test TEST_DATASET   path to testing dataset
  --batches BATCH_SIZE  number of samples to propogate (default: 64)
  --epochs NUM_EPOCHS   number of passes through dataset (default: 32)
  --save filename       save trained model (default "model.py")
  --disable-cuda        disable CUDA support
```
