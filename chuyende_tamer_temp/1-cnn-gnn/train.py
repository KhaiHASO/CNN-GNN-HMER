import sys
import os
import functools
import typing

# Monkeypatch for Python 3.7 compatibility with dagshub client
if not hasattr(functools, "cached_property"):
    class cached_property:
        def __init__(self, func):
            self.func = func
            self.__doc__ = func.__doc__
        def __get__(self, instance, owner):
            if instance is None:
                return self
            value = self.func(instance)
            instance.__dict__[self.func.__name__] = value
            return value
    functools.cached_property = cached_property

if not hasattr(typing, "Literal"):
    from typing_extensions import Literal
    typing.Literal = Literal

import wandb
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import CSVLogger

from tamer.datamodule import HMEDatamodule
from tamer.lit_tamer import LitTAMER

class DagsHubCallback(Callback):
    def __init__(self):
        super().__init__()
        self.last_uploaded_ckpt = None
        self.repo = os.environ.get("DAGSHUB_REPO")  # e.g., "KhaiHASO/CNN-GNN-HMER"
        self.token = os.environ.get("DAGSHUB_USER_TOKEN") or os.environ.get("DAGSHUB_TOKEN")
        self.branch = os.environ.get("DAGSHUB_BRANCH", "main")

    def get_loggers(self, trainer):
        if not hasattr(trainer, "logger") or trainer.logger is None:
            return []
        if hasattr(trainer.logger, "__iter__"):
            try:
                return list(trainer.logger)
            except Exception:
                return [trainer.logger]
        return [trainer.logger]

    def _get_repo_client(self):
        if not self.repo or not self.token:
            return None
        try:
            from dagshub.upload import Repo
            owner, repo_name = self.repo.split("/", 1)
            return Repo(owner, repo_name, token=self.token, branch=self.branch)
        except Exception as e:
            print(f"[DagsHub] Failed to initialize Repo client: {e}")
            return None

    def on_validation_end(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return
            
        repo_client = self._get_repo_client()
        if not repo_client:
            return

        # 1. Upload best checkpoint to DagsHub Storage Bucket (bucket=True)
        checkpoint_callback = None
        for cb in trainer.callbacks:
            if cb.__class__.__name__ == "ModelCheckpoint":
                checkpoint_callback = cb
                break
        
        if checkpoint_callback and checkpoint_callback.best_model_path:
            current_best = checkpoint_callback.best_model_path
            if current_best != self.last_uploaded_ckpt and os.path.exists(current_best):
                try:
                    print(f"[DagsHub] Uploading best checkpoint: {current_best}...")
                    repo_client.upload(
                        local_path=current_best,
                        remote_path="checkpoints/best_model.ckpt",
                        commit_message=f"Upload best checkpoint (val_ExpRate={trainer.callback_metrics.get('val_ExpRate', 0.0):.4f})",
                        bucket=True
                    )
                    self.last_uploaded_ckpt = current_best
                    print(f"[DagsHub] Successfully uploaded best checkpoint to DagsHub Storage.")
                except Exception as e:
                    print(f"[DagsHub] Failed to upload checkpoint: {e}")

        # 2. Upload metrics.csv to DagsHub Storage Bucket (bucket=True)
        for logger in self.get_loggers(trainer):
            if hasattr(logger, "log_dir") and logger.log_dir:
                metrics_csv = os.path.join(logger.log_dir, "metrics.csv")
                if os.path.exists(metrics_csv):
                    try:
                        repo_client.upload(
                            local_path=metrics_csv,
                            remote_path="metrics.csv",
                            commit_message="Upload metrics.csv",
                            bucket=True
                        )
                        print(f"[DagsHub] Successfully uploaded metrics.csv to DagsHub Storage.")
                    except Exception as e:
                        print(f"[DagsHub] Failed to upload metrics: {e}")

    def on_fit_end(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return
            
        repo_client = self._get_repo_client()
        if not repo_client:
            return

        # Upload final metrics.csv to the Git repository (bucket=False)
        # This will register the metrics in the DagsHub Experiments tab!
        for logger in self.get_loggers(trainer):
            if hasattr(logger, "log_dir") and logger.log_dir:
                metrics_csv = os.path.join(logger.log_dir, "metrics.csv")
                if os.path.exists(metrics_csv):
                    try:
                        print(f"[DagsHub] Uploading final metrics.csv to Git repository (branch: {self.branch})...")
                        repo_client.upload(
                            local_path=metrics_csv,
                            remote_path="metrics.csv",
                            commit_message="Log final training metrics from training run",
                            bucket=False
                        )
                        print(f"[DagsHub] Successfully logged final metrics to DagsHub Experiments.")
                    except Exception as e:
                        print(f"[DagsHub] Failed to upload final metrics to Git: {e}")

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
            "callbacks": [DagsHubCallback()]
        },
    )

if __name__ == "__main__":
    cli_main()