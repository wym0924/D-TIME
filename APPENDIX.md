## Complete Configuration Parameter Description

| Parameter Category   | Parameter Name      | Description                          | Example Value  |
|----------------------|---------------------|--------------------------------------|----------------|
| Model Hyperparameters| seq_len             | Lookback Length (Input Sequence)     | 512            |
| Model Hyperparameters| batch_size          | Batch Size                           | 128            |
| Model Hyperparameters| lr                  | Learning Rate                        | 0.0005         |
| Model Hyperparameters| weight_decay        | Weight Decay                         | 0.0001         |
| Training Configuration| num_epochs        | Number of Training Epochs            | 100            |
| Training Configuration| patience          | Early Stopping Patience              | 5              |
| Evaluation Configuration| seed            | Random Seed                          | 2021           |
| Hardware Configuration| gpu              | GPU ID Used                          | 0              |
| Hardware Configuration| gpu_driver_version | GPU Driver Version               | 525.105.17     |

### Early Stopping Strategy Description
The early stopping strategy is designed to prevent model overfitting. Training is terminated when the validation set loss fails to improve over consecutive epochs. The core parameters are as follows:

| Parameter Name   | Meaning                                                                                                 | Example Value |
|------------------|---------------------------------------------------------------------------------------------------------|---------------|
| `patience`       | The maximum number of epochs (patience value) allowed for the validation loss to remain unchanged (non-decreasing); training stops once this threshold is exceeded | 5             |
| `early_stop_delta` | The minimum threshold for loss reduction; if the magnitude of loss reduction is ≤ this value, it is deemed that no significant improvement has been achieved | 0 / 0.001     |

**Strategy Logic**:  
- The loss is calculated after each validation epoch. If the magnitude of loss reduction exceeds `delta`, the optimal model is updated and the counter is reset;  
- If the loss shows no significant improvement for `patience` consecutive epochs, early stopping is triggered, and the parameters of the optimal model are adopted as the final result.

**Configuration for Each Dataset**:  
All datasets uniformly use `patience=5` and `early_stop_delta=0` (no minimum threshold) to ensure stable convergence of the training process.
