
from pytorch_tabnet.callbacks import Callback

import wandb


class PrintLR(Callback):
    def __init__(self):
        self.previous_lr = None

    def on_epoch_end(self, epoch, logs=None):
        current_lr = self.trainer.optimizer_params['lr']
        if (
            self.previous_lr is None or
            current_lr != self.previous_lr
        ):
            print(f"Epoch {epoch+1}: "+
                  f"Learning rate changed to "+
                  f"{current_lr:.6f}")
        self.previous_lr = current_lr


class WandbCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        # logs = self.trainer.history.epoch_metrics
        wandb.log({"epoch": epoch + 1, **logs})

