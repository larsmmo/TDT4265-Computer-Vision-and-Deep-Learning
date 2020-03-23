# Evaluation of models on the TDT4265 Dataset

To run evaluation on the dataset, we've created an automatic python script for you:

Run `submit_results.py` with the config file to your model, for example:
```bash
python3 submit_results.py configs/train_tdt4265.yml
```

This will load the most recent checkpoint and perform inference on the entire tdt4265 test dataset.

Note that the mAP result you get is not the result on the entire dataset (only 30%). The mAP for the entire dataset will be public after the project submission deadline.

