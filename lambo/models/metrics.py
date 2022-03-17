import torch


def quantile_calibration(mean, std, targets):
    quantiles = torch.linspace(0.05, 0.95, 10, device=mean.device).view(10, 1, 1)

    z_dist = torch.distributions.Normal(
        torch.tensor((0.0,), device=mean.device),
        torch.tensor((1.0,), device=mean.device),
    )
    tail_probs = (1 - quantiles) / 2
    z_scores = z_dist.icdf(1 - tail_probs)  # (num_quantiles, 1, 1)

    pred_mean = mean.unsqueeze(0)  # (1, batch_size, target_dim)
    pred_std = std.unsqueeze(0)
    lb = pred_mean - z_scores * pred_std
    ub = pred_mean + z_scores * pred_std

    targets = targets.unsqueeze(0)
    targets_in_region = torch.le(lb, targets) * torch.le(targets, ub)
    occupancy_rates = targets_in_region.float().mean(-1, keepdim=True)  # average over target dim
    occupancy_rates = occupancy_rates.mean(-2, keepdim=True)  # average over batch dim
    # import pdb; pdb.set_trace()
    ece = (occupancy_rates - quantiles).abs().mean().item()
    calibration_metrics = {
        f"{quantile.item():.2f}_quantile": occ_rate.item()
        for quantile, occ_rate in zip(quantiles, occupancy_rates)
    }
    calibration_metrics["ece"] = ece
    # in general,
    # over-confident --> quantiles > occupancy rates --> positive diff
    # under-confident --> quantiles < occupancy rates --> negative diff
    calibration_metrics["occ_diff"] = (quantiles - occupancy_rates).mean().item()
    return calibration_metrics
