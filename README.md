# Sharpened Cosine Similarity
Original code from: https://github.com/brohrer/sharpened_cosine_similarity_torch. See the original repository for a more comprehensive readme.

Changes:
- Code uses CUDA capable device if available
- Added tensorboard logging
- Huge refactoring
- Uses torch.jit.script on the module. This gave at least 30% improvement on GPU runtime of the original model.
- Uses the optimized implementation from jatentaki: https://github.com/brohrer/sharpened_cosine_similarity_torch/pull/6

Feel free to copy any parts of this code.

### CIFAR10 results (over 1 run).

Model| Num params | CIFAR10 Test accuracy
---|---|---|
Original | 68K | 83.0%
Revised | 1.2M | 89.9%
Revised+residual | 1.25M | 91.3%

View the tensorboard logs here:
https://tensorboard.dev/experiment/AY27LbxrRpaBHNMO0m9Wkw/

### Reproduce:
Original model from brohrer:
```
python train.py run_name --model original
```

Revised architecture:
```
python train.py run_revised --model revised
```

Revised architecture with residual connections:
```
python train.py revised_residual --model revised_residual
```
