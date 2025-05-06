

# Data Processing 
- How is the image been loaded ->  processed
- How 
Now As I have all these, write a trainer script. 
- Use Huggingface Trainer API to train the model.
- Include logging with wandb.
- Use the following hyperparameters. These should be configurable via a config class or a dictionary or file.
  - learning_rate: 5e-5
  - num_train_epochs: 10
  - per_device_train_batch_size: 2
  - per_device_eval_batch_size: 2
  - weight_decay: 0
  - lr_scheduler_type: cosine
  - warmup_steps: 5000
  - (if can prefetch the data, do it)
  - logging_steps: 10
Always set the seed to 42 for reproducibility!
For modle, use swin transformer from segmentation models pytorch. don't use the pretrained weights.
