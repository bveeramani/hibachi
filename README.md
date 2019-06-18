## Usage
```
usage: train.py [-h] [--test TEST_DATASET] [--batches BATCH_SIZE]
                [--epochs NUM_EPOCHS] [--save FILENAME] [--disable-cuda]
                train

Train model

positional arguments:
  train                 path to trainining dataset

optional arguments:
  -h, --help            show this help message and exit
  --test TEST_DATASET   path to testing dataset
  --batches BATCH_SIZE  number of samples to propogate (default: 64)
  --epochs NUM_EPOCHS   number of passes through dataset (default: 32)
  --save FILENAME       save trained model (default "model.py")
  --disable-cuda        disable CUDA support
```

## Example
```
from train import train, save
from models import LogisticRegressionModel
from datasets import OteyP450

dataset = OteyP450("OteyP450/processed/train.csv")
model = LogisticRegressionModel(8)
model = train(model, dataset)
save(model, "logistic.pt")

from explainers import OteyP450LimeExplainer

local_explain = OteyP450LimeExplainer(model, dataset)
arbitrary_sample = dataset[83]
local_explain(arbitrary_sample)

from explainers import CCMExplainer

global_explain = CCMExplainer(4)
global_explain(dataset)
```
