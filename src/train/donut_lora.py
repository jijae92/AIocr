"""
Donut LoRA Fine-tuning Script.

Fine-tune Donut model with LoRA (Low-Rank Adaptation) using:
- Accelerate for distributed training
- PEFT (Parameter-Efficient Fine-Tuning) for LoRA
- SynthDoG dataset or custom dataset

Designed for cloud GPU training (1+ GPU), with adapter
loading support for macOS MPS inference.

Usage:
    python src/train/donut_lora.py \
        --config configs/train_donut.yaml \
        --data-dir data/train \
        --output-dir models/donut_lora

Requirements:
    - accelerate
    - peft
    - transformers
    - torch
    - datasets
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    get_scheduler,
)

from train.synthdog_loader import SynthDogDataset, custom_collate_fn
from util.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Model
    model_name: str = "naver-clova-ix/donut-base"
    max_length: int = 1024

    # LoRA
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )

    # Training
    num_epochs: int = 10
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    warmup_steps: int = 100
    max_grad_norm: float = 1.0

    # Data
    train_data_dir: str = "data/train"
    val_data_dir: Optional[str] = None
    image_size: tuple = (1280, 960)

    # Output
    output_dir: str = "models/donut_lora"
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 50

    # Device
    fp16: bool = True
    gradient_checkpointing: bool = True

    @classmethod
    def from_yaml(cls, config_path: Path) -> "TrainingConfig":
        """Load config from YAML file."""
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Extract training section
        training_config = config_dict.get("training", {})

        return cls(**training_config)


class DonutLoRATrainer:
    """Donut LoRA trainer with Accelerate."""

    def __init__(
        self,
        config: TrainingConfig,
        accelerator: Optional[Accelerator] = None,
    ):
        """
        Initialize trainer.

        Args:
            config: Training configuration
            accelerator: Accelerate accelerator instance
        """
        self.config = config
        self.accelerator = accelerator or Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision="fp16" if config.fp16 else "no",
        )

        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.processor = None
        self.model = None
        self.optimizer = None
        self.scheduler = None

        # Training state
        self.global_step = 0
        self.best_val_loss = float("inf")

    def setup_model(self):
        """Setup Donut model with LoRA."""
        logger.info(f"Loading Donut model: {self.config.model_name}")

        # Load processor
        self.processor = DonutProcessor.from_pretrained(self.config.model_name)

        # Load base model
        base_model = VisionEncoderDecoderModel.from_pretrained(
            self.config.model_name
        )

        # Enable gradient checkpointing for memory efficiency
        if self.config.gradient_checkpointing:
            base_model.encoder.gradient_checkpointing_enable()

        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
        )

        # Apply LoRA
        self.model = get_peft_model(base_model, lora_config)
        self.model.print_trainable_parameters()

        logger.info("LoRA model setup complete")

    def setup_data(self) -> tuple:
        """
        Setup dataloaders.

        Returns:
            (train_loader, val_loader)
        """
        logger.info("Setting up datasets")

        # Training dataset
        train_dataset = SynthDogDataset(
            data_dir=Path(self.config.train_data_dir),
            processor=self.processor,
            max_length=self.config.max_length,
            image_size=self.config.image_size,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=custom_collate_fn,
            num_workers=4,
            pin_memory=True,
        )

        # Validation dataset (optional)
        val_loader = None
        if self.config.val_data_dir:
            val_dataset = SynthDogDataset(
                data_dir=Path(self.config.val_data_dir),
                processor=self.processor,
                max_length=self.config.max_length,
                image_size=self.config.image_size,
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=custom_collate_fn,
                num_workers=4,
                pin_memory=True,
            )

        logger.info(f"Training samples: {len(train_dataset)}")
        if val_loader:
            logger.info(f"Validation samples: {len(val_dataset)}")

        return train_loader, val_loader

    def setup_optimizer(self, num_training_steps: int):
        """Setup optimizer and scheduler."""
        # AdamW optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )

        # Linear scheduler with warmup
        self.scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_training_steps,
        )

    def train(self):
        """Run training loop."""
        logger.info("Starting Donut LoRA training")

        # Setup
        self.setup_model()
        train_loader, val_loader = self.setup_data()

        # Calculate training steps
        num_training_steps = (
            len(train_loader)
            * self.config.num_epochs
            // self.config.gradient_accumulation_steps
        )

        self.setup_optimizer(num_training_steps)

        # Prepare with Accelerate
        (
            self.model,
            self.optimizer,
            train_loader,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.model, self.optimizer, train_loader, self.scheduler
        )

        if val_loader:
            val_loader = self.accelerator.prepare(val_loader)

        # Training loop
        progress_bar = tqdm(
            total=num_training_steps,
            disable=not self.accelerator.is_local_main_process,
        )

        for epoch in range(self.config.num_epochs):
            self.model.train()
            epoch_loss = 0.0

            for batch_idx, batch in enumerate(train_loader):
                with self.accelerator.accumulate(self.model):
                    # Forward pass
                    outputs = self.model(**batch)
                    loss = outputs.loss

                    # Backward pass
                    self.accelerator.backward(loss)

                    # Gradient clipping
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.max_grad_norm,
                        )

                    # Optimizer step
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    epoch_loss += loss.item()

                    # Update progress
                    if self.accelerator.sync_gradients:
                        progress_bar.update(1)
                        self.global_step += 1

                        # Logging
                        if self.global_step % self.config.logging_steps == 0:
                            avg_loss = epoch_loss / (batch_idx + 1)
                            lr = self.optimizer.param_groups[0]["lr"]

                            if self.accelerator.is_local_main_process:
                                logger.info(
                                    f"Step {self.global_step}: "
                                    f"loss={avg_loss:.4f}, lr={lr:.2e}"
                                )

                        # Evaluation
                        if (
                            val_loader
                            and self.global_step % self.config.eval_steps == 0
                        ):
                            val_loss = self.evaluate(val_loader)
                            logger.info(f"Validation loss: {val_loss:.4f}")

                            # Save best model
                            if val_loss < self.best_val_loss:
                                self.best_val_loss = val_loss
                                self.save_checkpoint("best")

                        # Save checkpoint
                        if self.global_step % self.config.save_steps == 0:
                            self.save_checkpoint(f"step_{self.global_step}")

            # End of epoch
            avg_epoch_loss = epoch_loss / len(train_loader)
            logger.info(
                f"Epoch {epoch + 1}/{self.config.num_epochs}: "
                f"avg_loss={avg_epoch_loss:.4f}"
            )

            # Save epoch checkpoint
            self.save_checkpoint(f"epoch_{epoch + 1}")

        # Save final model
        self.save_checkpoint("final")
        logger.info("Training complete!")

    def evaluate(self, val_loader) -> float:
        """
        Evaluate on validation set.

        Args:
            val_loader: Validation dataloader

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
                num_batches += 1

        self.model.train()
        return total_loss / num_batches if num_batches > 0 else 0.0

    def save_checkpoint(self, checkpoint_name: str):
        """
        Save model checkpoint.

        Args:
            checkpoint_name: Name for checkpoint
        """
        if not self.accelerator.is_local_main_process:
            return

        checkpoint_dir = self.output_dir / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Unwrap model
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        # Save LoRA adapter
        unwrapped_model.save_pretrained(checkpoint_dir)

        # Save processor
        self.processor.save_pretrained(checkpoint_dir)

        # Save training state
        state = {
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "config": self.config.__dict__,
        }

        with open(checkpoint_dir / "training_state.json", "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"Saved checkpoint: {checkpoint_dir}")


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Donut LoRA Fine-tuning")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_donut.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Training data directory (overrides config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for model (overrides config)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use mixed precision training",
    )

    args = parser.parse_args()

    # Load config
    config = TrainingConfig.from_yaml(Path(args.config))

    # Override with command line args
    if args.data_dir:
        config.train_data_dir = args.data_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.fp16:
        config.fp16 = True

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision="fp16" if config.fp16 else "no",
    )

    # Initialize trainer
    trainer = DonutLoRATrainer(config=config, accelerator=accelerator)

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
