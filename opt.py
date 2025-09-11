import torch

from metrics import compute_metrics
from utils import save_eval_images, save_sample_images
from logger import MetricLogger, SmoothedValue


def train_one_epoch(
    args,
    model,
    data_loader,
    optimizer,
    loss_fn,
    epoch,
    log_file,
    print_freq=10,
    eval_in_train=False,
):
    model.train()

    metric_logger = MetricLogger(delimiter="  ", log_file=log_file)
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Train epoch: [{epoch}]"

    for batch_idx, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        inputs = batch["inputs"].to(args.device)
        targets = batch["targets"].to(args.device)
        if inputs is None:
            raise ValueError("inputs is None")
        if targets is None:
            raise ValueError("targets is None")
        pred_l = model(inputs)

        loss = loss_fn(pred_l, targets)
        total_loss = loss["total"]

        optimizer.zero_grad()
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        for loss_name, loss_value in loss.items():
            metric_logger.update(**{f"{loss_name}_loss": loss_value.item()})

        for param_group in optimizer.param_groups:
            metric_logger.update(lr=param_group["lr"])

        # For logging/metrics and image saving, clamp to [0,1]
        pred_vis = pred_l.clamp(0.0, 1.0)

        if eval_in_train:
            metrics = compute_metrics(
                targets,
                pred_vis,
                str(args.device),
                scale=args.scale,
                crop_border=True,
                y_channel=True,
            )
            for metric_name, metric_value in metrics.items():
                metric_logger.update(**{f"{metric_name}": metric_value})

        if batch_idx % (print_freq * 5) == 0:
            save_sample_images(
                inputs, pred_vis, targets, batch_idx, epoch, args.output_dir
            )

    metric_logger.synchronize_between_processes()
    print(f"Train stats: {metric_logger}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate_fn(
    args,
    data_loader,
    model,
    epoch,
    log_file,
    print_freq=100,
):
    model.eval()

    metric_logger = MetricLogger(delimiter="  ", log_file=log_file)
    header = f"Test: [{epoch}]"

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)
        ):
            inputs = batch["inputs"].to(args.device)
            targets = batch["targets"].to(args.device)
            filenames = batch["filenames"]

            pred_l = model(inputs)
            pred_vis = pred_l.clamp(0.0, 1.0)

            metrics = compute_metrics(
                targets,
                pred_vis,
                str(args.device),
            )

            for metric_name, metric_value in metrics.items():
                metric_logger.update(**{f"{metric_name}": metric_value})

            if args.save_images:
                save_eval_images(
                    inputs, pred_vis, targets, filenames, epoch, args.output_dir
                )

    metric_logger.synchronize_between_processes()
    print(f"Test stats: {metric_logger}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
