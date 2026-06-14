import sys
import os
import wandb
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import MLFlowLogger

from tamer.datamodule import HMEDatamodule
from tamer.lit_tamer import LitTAMER

class MLflowCheckpointCallback(Callback):
    def __init__(self):
        super().__init__()
        self.last_uploaded_ckpt = None

    def on_validation_end(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return
            
        checkpoint_callback = None
        for cb in trainer.callbacks:
            if cb.__class__.__name__ == "ModelCheckpoint":
                checkpoint_callback = cb
                break
        
        if checkpoint_callback and checkpoint_callback.best_model_path:
            current_best = checkpoint_callback.best_model_path
            if current_best != self.last_uploaded_ckpt and os.path.exists(current_best):
                loggers = trainer.loggers if hasattr(trainer, "loggers") else [trainer.logger]
                for logger in loggers:
                    if isinstance(logger, MLFlowLogger):
                        try:
                            run_id = logger.run_id
                            if run_id:
                                print(f"[MLflow] Uploading best checkpoint to DagsHub MLflow: {current_best}...")
                                logger.experiment.log_artifact(
                                    run_id=run_id,
                                    local_path=current_best,
                                    artifact_path="checkpoints"
                                )
                                self.last_uploaded_ckpt = current_best
                                print("[MLflow] Successfully uploaded checkpoint.")
                            else:
                                print("[MLflow] Warning: run_id is empty, cannot upload checkpoint.")
                        except Exception as e:
                            print(f"[MLflow] Failed to upload checkpoint: {e}")

def cli_main():
    # Handle WandB API Key from CLI (e.g., --wandb_api_key=XYZ or --wandb_api_key XYZ)
    if "--wandb_api_key" in sys.argv:
        try:
            idx = sys.argv.index("--wandb_api_key")
            key = sys.argv[idx + 1]
            wandb.login(key=key)
            print(f"Logged in to WandB with provided key.")
            del sys.argv[idx]
            del sys.argv[idx] 
        except IndexError:
            print("Error: --wandb_api_key flag provided but no key found.")
    
    # Also support --wandb_api_key=XYZ format
    for i, arg in enumerate(sys.argv):
        if arg.startswith("--wandb_api_key="):
            key = arg.split("=", 1)[1]
            wandb.login(key=key)
            print(f"Logged in to WandB with provided key.")
            del sys.argv[i]
            break

    cli = LightningCLI(
        LitTAMER,
        HMEDatamodule,
        save_config_overwrite=True,
        trainer_defaults={
            "plugins": DDPPlugin(find_unused_parameters=True),
            "callbacks": [MLflowCheckpointCallback()]
        },
    )

if __name__ == "__main__":
    cli_main()