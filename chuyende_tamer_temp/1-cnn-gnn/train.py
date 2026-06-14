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
    def on_validation_end(self, trainer, pl_module):
        checkpoint_callback = None
        for cb in trainer.callbacks:
            if cb.__class__.__name__ == "ModelCheckpoint":
                checkpoint_callback = cb
                break
        
        if checkpoint_callback and checkpoint_callback.best_model_path:
            if os.path.exists(checkpoint_callback.best_model_path):
                for logger in trainer.loggers:
                    if isinstance(logger, MLFlowLogger):
                        try:
                            logger.experiment.log_artifact(
                                run_id=logger.run_id,
                                local_path=checkpoint_callback.best_model_path,
                                artifact_path="checkpoints"
                            )
                            print(f"[MLflow] Automatically uploaded best checkpoint: {checkpoint_callback.best_model_path}")
                        except Exception as e:
                            print(f"[MLflow] Failed to upload checkpoint: {e}")

def cli_main():
    # Handle WandB API Key from CLI (e.g., --wandb_api_key=XYZ or --wandb_api_key XYZ)
    # This allows passing the key without it being a valid LightningCLI argument
    if "--wandb_api_key" in sys.argv:
        try:
            idx = sys.argv.index("--wandb_api_key")
            key = sys.argv[idx + 1]
            wandb.login(key=key)
            print(f"Logged in to WandB with provided key.")
            # Remove arguments to prevent LightningCLI from crashing
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