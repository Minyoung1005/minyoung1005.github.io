---
layout: post
title: "Saving and Loading Models"
categories: [PyTorch]
---

```
import torch

# Initialize model and optimizer

model = ModelClass(*args, **kwargs)
optimizer = OptimizerClass(*args, **kwargs)

# load checkpoint
checkpoint=torch.load_state_dict('YOUR_CHECKPOINT_PATH')

# load states from checkpoint
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])

# If mismatching keys exist, add strict=False as an addtional argument for load_state_dict()

# If you're just proceeding a temporarily paused network training, 
# load other infos such as epoch, loss, etc.
epoch = checkpoint['epoch']
loss = checkpoint['loss']
exp_dir = checkpoint['exp_dir']
args = checkpoint['args']

# After loading the model, train or evaluate the network
model.eval()
or
model.train()

```

```
# Save model weights and informations as checkpoint

torch.save({
            'model': model.state_dict()},
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss,
            'args': args,
            ...
            },'PATH')
```

### When do we need to load optimizer?


