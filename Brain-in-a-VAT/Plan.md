

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

The loss is just the simple MSE loss with a coefficient from DiffusionScheduler.

SO the data set will give a source image, and a target image.
A random time step t is sampled for the diffusion scheduler.
using this time step, the DiffusionScheduler can be used to generate a noisy image from the source image and the target image.
the noise image is the input to the model.
The target image is the output of the model.
we get a loss coefficient from the DiffusionScheduler, and we use this to scale the loss. the loss is just the MSE loss between the model output and the target image.
The model output is the fully denoised image.

The config = CFG(kappa=2.0, p=0.5, eta_T=0.999, T=50) 

No augumentation, do use accelerate to prepare the model no need for VAE (but leave option to use / add VAE) use torch.compile to prepare the model if possible to make it faster, evalutiaon should be per epoch, the save stratehy is also the best of evaluation.load_best_model_at_end=True (fp16=True) and report_to="wandb"
no early sopping DO all the seeding
