
import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from dataset import make_train_val_datasets
from ai_weather_model import NowcastNet, weighted_rain_mse


def parse_args():
    p = argparse.ArgumentParser()

    # Path configuration
    p.add_argument("--dynamic-dir", default="data/training/dynamic_layers")
    p.add_argument("--static-dir", default="data/training/static_layers")
    p.add_argument("--checkpoint-dir", default="checkpoints")

    # Model and slice architecture
    p.add_argument("--t-in", type=int, default=4, help="number of input frames")
    p.add_argument("--t-out", type=int, default=6, help="number of frames to forecast")
    p.add_argument("--patch-size", type=int, default=128)
    p.add_argument("--samples-per-event", type=int, default=4,
                    help="random (window, crop) samples drawn per event per epoch")
    
    # Training hyperparameters
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-channels", type=int, nargs="+", default=[32, 64, 64])
    p.add_argument("--weight-power", type=float, default=2.0,
                    help="exponent for upweighting heavy-rain pixels in the loss")
    
    # Advection Prior toggles (True by default)
    p.add_argument("--use-advection-prior", action="store_true", default=True,
                    help="use optical-flow extrapolation as a physics baseline, "
                         "with the network learning a residual correction (recommended)")
    p.add_argument("--no-advection-prior", dest="use_advection_prior", action="store_false")
    p.add_argument("--advection-damping", type=float, default=0.0,
                    help="per-step intensity decay in the advection extrapolation, 0.0-0.05 typical")
    
    # Hardware/Data pipeline efficiency
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--resume", default=None, help="path to a checkpoint to resume from")

    # Automatic Mixed Precision (AMP) toggles
    p.add_argument("--amp", action="store_true", default=True,
                    help="use mixed precision (recommended, on by default)")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    return p.parse_args()


def run_epoch(model, loader, device, optimizer=None, weight_power=2.0, amp=False, scaler=None):
    """One pass over `loader`. If optimizer is given, trains; otherwise
    evaluates in no_grad mode. Returns the mean loss over the epoch."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, n_batches = 0.0, 0
    for x, static, y, prior in loader:
        # Push tensors to the targeted hardware device (CPU or CUDA GPU)
        x, static, y, prior = x.to(device), static.to(device), y.to(device), prior.to(device)
        t_out = y.shape[1]

        # Context manager toggles gradient calculation to save memory during evaluation
        with torch.set_grad_enabled(is_train):
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                pred = model(x, static, forecast_steps=t_out, advection_prior=prior)
                loss = weighted_rain_mse(pred, y, weight_power=weight_power)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Hardware detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # autotune conv algorithms; safe since
        # every batch has the same fixed patch_size/t_in/t_out shape
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)}, "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    else:
        print(f"Using device: {device}")
    if device.type == "cpu":
        print("WARNING: no GPU found. ConvLSTM training on CPU will be very slow -- "
              "this is fine for a smoke test with small patch-size/batch-size, "
              "but for a real training run you'll want a CUDA GPU.")

    # Initialises dataset
    train_ds, val_ds = make_train_val_datasets(
        dynamic_dir=args.dynamic_dir,
        static_dir=args.static_dir,
        t_in=args.t_in,
        t_out=args.t_out,
        patch_size=args.patch_size,
        samples_per_event=args.samples_per_event,
        val_fraction=args.val_fraction,
        use_advection_prior=args.use_advection_prior,
        advection_damping=args.advection_damping,
    )
    print(f"Train events: {len(train_ds.event_files)}  Val events: {len(val_ds.event_files)}")

    # Configures DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    # Instantiates Model
    model = NowcastNet(
        dynamic_channels=2,
        static_channels=3,
        hidden_channels=tuple(args.hidden_channels),
        forecast_steps=args.t_out,
    ).to(device)

    # Optimizer, Scheduler, and AMP Setup
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Setups GradScaler to support AMP
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    # Checkpoint Resuming
    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    # Epoch Loop
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # Train pass
        train_loss = run_epoch(
            model, train_loader, device, optimizer=optimizer,
            weight_power=args.weight_power, amp=args.amp, scaler=scaler,
        )

        # Validation pass
        val_loss = run_epoch(
            model, val_loader, device, optimizer=None,
            weight_power=args.weight_power, amp=args.amp,
        )

        # Steps Learning Rate scheduler based on validation performance
        scheduler.step(val_loss)
        dt = time.time() - t0

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.10f}  "
              f"val_loss={val_loss:.10f}  lr={current_lr:.2e}  ({dt:.1f}s)")

        # Saves standard checkpoints
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "args": vars(args),
        }
        torch.save(checkpoint, os.path.join(args.checkpoint_dir, "last.pt"))

        # Saves best model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint["best_val_loss"] = best_val_loss
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, "best.pt"))
            print(f"  -> new best model saved (val_loss={val_loss:.10f})")


if __name__ == "__main__":
    main()