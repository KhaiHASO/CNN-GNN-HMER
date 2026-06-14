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

    def _convert_metrics(self, trainer, pl_metrics_csv, target_csv):
        import csv
        import time
        if not os.path.exists(pl_metrics_csv):
            return False
        try:
            rows = []
            with open(pl_metrics_csv, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(row)
            
            timestamp = int(time.time() * 1000)
            with open(target_csv, "w", newline="") as f:
                writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
                f.write("Name,Value,Timestamp,Step\n")
                for row in rows:
                    step = row.get("step")
                    epoch = row.get("epoch")
                    step_num = 1
                    if step is not None and step != "":
                        step_num = int(step)
                    elif epoch is not None and epoch != "":
                        step_num = int(epoch)
                    
                    for k, v in row.items():
                        if k in ["step", "epoch"] or v is None or v == "":
                            continue
                        try:
                            val = float(v)
                            writer.writerow([k, val, timestamp, step_num])
                        except ValueError:
                            continue
            return True
        except Exception as e:
            print(f"[DagsHub] Failed to convert metrics: {e}")
            return False

    def _write_params(self, trainer, target_yml):
        try:
            import yaml
            hparams = {}
            if hasattr(trainer, "lightning_module") and hasattr(trainer.lightning_module, "hparams"):
                hparams.update(trainer.lightning_module.hparams)
            
            serializable_hparams = {}
            for k, v in hparams.items():
                if type(v) in [int, float, bool, str, list, dict]:
                    serializable_hparams[k] = v
                else:
                    serializable_hparams[k] = str(v)
            
            with open(target_yml, "w") as f:
                yaml.safe_dump(serializable_hparams, f)
            return True
        except Exception as e:
            print(f"[DagsHub] Failed to write params.yml: {e}")
            return False

    def _upload_logs(self, repo_client, trainer, pl_metrics_csv, bucket=False):
        import shutil
        import tempfile
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        try:
            # 1. Convert metrics
            root_metrics_csv = os.path.join(temp_dir, "metrics.csv")
            has_metrics = self._convert_metrics(trainer, pl_metrics_csv, root_metrics_csv)
            
            # 2. Write parameters
            root_params_yml = os.path.join(temp_dir, "params.yml")
            has_params = self._write_params(trainer, root_params_yml)
            
            # 3. Upload directory if we have files
            if has_metrics or has_params:
                print(f"[DagsHub] Uploading metrics.csv and params.yml to {'Storage' if bucket else 'Git (branch: ' + self.branch + ')'}...")
                upload_kwargs = {
                    "local_path": temp_dir,
                    "remote_path": "",  # Upload at the root level of the repo/bucket
                    "commit_message": "Log training metrics and hyperparameters",
                    "bucket": bucket
                }
                if not bucket:
                    upload_kwargs["versioning"] = "git"
                    upload_kwargs["force"] = True
                repo_client.upload(**upload_kwargs)
                print(f"[DagsHub] Successfully uploaded training logs.")
        except Exception as e:
            print(f"[DagsHub] Failed to upload training logs: {e}")
        finally:
            shutil.rmtree(temp_dir)

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

        # 2. Convert and Upload metrics.csv and params.yml to DagsHub Storage Bucket (bucket=True)
        for logger in self.get_loggers(trainer):
            if hasattr(logger, "log_dir") and logger.log_dir:
                pl_metrics_csv = os.path.join(logger.log_dir, "metrics.csv")
                if os.path.exists(pl_metrics_csv):
                    self._upload_logs(repo_client, trainer, pl_metrics_csv, bucket=True)

    def on_fit_end(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return
            
        repo_client = self._get_repo_client()
        if not repo_client:
            return

        # Upload final metrics.csv and params.yml to the Git repository (bucket=False)
        # This will register the metrics and params in the DagsHub Experiments tab!
        for logger in self.get_loggers(trainer):
            if hasattr(logger, "log_dir") and logger.log_dir:
                pl_metrics_csv = os.path.join(logger.log_dir, "metrics.csv")
                if os.path.exists(pl_metrics_csv):
                    self._upload_logs(repo_client, trainer, pl_metrics_csv, bucket=False)

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