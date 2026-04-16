#!/usr/bin/env python3
"""
DRIVE deep-supervision lambda grid evaluation.

Examples:
  python eval_drive.py --models_dir ./models/drive --data_dir ./data/DRIVE --results_dir ./results/drive
  python eval_drive.py --models_dir ./models/drive --data_dir ./data/DRIVE --results_dir ./results/drive --optimize_threshold
"""

import os
import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix, matthews_corrcoef

from improved_pdc_convolutions import normalize_config_string
from pdc_unet_model import create_model
from drive_preprocessing import get_test_loader, predict_full_image
from path_compat import apply_model_path_aliases


def safe_convert_for_json(obj):
    import numpy as _np
    import pandas as _pd

    if isinstance(obj, dict):
        return {k: safe_convert_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_convert_for_json(v) for v in obj]
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.floating,)):
        return float(obj)
    if isinstance(obj, (_np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (_pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, (_pd.Series, _pd.DataFrame)):
        try:
            return obj.to_dict()
        except Exception:
            return str(obj)
    return str(obj)


def parse_float_list(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip() != ""]


def parse_int_list(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


def estimate_flops_g(model: nn.Module, input_size=(1, 1, 48, 48), device="cuda") -> float:
    try:
        from thop import profile
    except Exception:
        return float("nan")

    try:
        dev = device if isinstance(device, torch.device) else torch.device(device)
        dummy = torch.randn(*input_size, device=dev)
        model.eval()
        with torch.no_grad():
            flops, _ = profile(model, inputs=(dummy,), verbose=False)
        return float(flops) / 1e9
    except Exception:
        return float("nan")


def _fmt_metric(v, digits=4):
    try:
        x = float(v)
    except Exception:
        return "nan"
    if np.isnan(x) or np.isinf(x):
        return "nan"
    return f"{x:.{digits}f}"


def print_terminal_metrics(row):
    print("  Acc\tSe\tSp\tF1\tAUC")
    print(
        "  {}\t{}\t{}\t{}\t{}".format(
            _fmt_metric(row.get("Accuracy", float("nan")), 4),
            _fmt_metric(row.get("Sensitivity", float("nan")), 4),
            _fmt_metric(row.get("Specificity", float("nan")), 4),
            _fmt_metric(row.get("F1", float("nan")), 4),
            _fmt_metric(row.get("AUC", float("nan")), 4),
        )
    )
    print("  Params(M)\tFLOPs(G)\tFPS\tTime(ms)")
    print(
        "  {}\t{}\t{}\t{}".format(
            _fmt_metric(row.get("Params(M)", float("nan")), 4),
            _fmt_metric(row.get("FLOPs(G)", float("nan")), 4),
            _fmt_metric(row.get("FPS", float("nan")), 4),
            _fmt_metric(row.get("Time(ms)", float("nan")), 4),
        )
    )


class WrappedDeepSupervisionModel(nn.Module):
    """Expose only the primary prediction for patch-based inference."""

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        out = self.base_model(x)
        if isinstance(out, (list, tuple)):
            return out[0]
        return out


class LambdaGridEvaluator:
    def __init__(
        self,
        device="cuda",
        fixed_config="C-[V]脳11",
        fixed_channels=32,
        threshold=0.40,
        optimize_threshold=False,
        thr_min=0.30,
        thr_max=0.60,
        thr_step=0.01,
        f1_drop_tolerance=0.0005,
    ):
        self.device = device
        try:
            self.fixed_config = normalize_config_string(fixed_config)
        except ValueError:
            self.fixed_config = fixed_config
        self.fixed_channels = fixed_channels
        self.threshold = threshold
        self.optimize_threshold = optimize_threshold
        self.thr_min = thr_min
        self.thr_max = thr_max
        self.thr_step = thr_step
        self.f1_drop_tolerance = f1_drop_tolerance

    def build_model(
        self,
        config_str=None,
        channels=None,
        use_gpdc=True,
        use_residual=False,
        use_lmm=False,
        use_sdpm=True,
        use_deep_supervision=True,
    ):
        return create_model(
            config_str=self.fixed_config if config_str is None else config_str,
            channels=self.fixed_channels if channels is None else int(channels),
            use_gpdc=bool(use_gpdc),
            use_residual=bool(use_residual),
            use_lmm=bool(use_lmm),
            use_sdpm=bool(use_sdpm),
            use_deep_supervision=bool(use_deep_supervision),
        ).to(self.device)

    def _infer_model_kwargs(self, checkpoint):
        model_kwargs = {
            "config_str": self.fixed_config,
            "channels": self.fixed_channels,
            "use_gpdc": True,
            "use_residual": False,
            "use_lmm": False,
            "use_sdpm": True,
            "use_deep_supervision": True,
        }
        if not isinstance(checkpoint, dict):
            return model_kwargs

        args_info = checkpoint.get("args", {})
        args_info = args_info if isinstance(args_info, dict) else {}
        experiment_info = checkpoint.get("experiment_info", {})
        experiment_info = experiment_info if isinstance(experiment_info, dict) else {}

        model_kwargs["config_str"] = checkpoint.get(
            "config",
            experiment_info.get("config", args_info.get("fixed_config", model_kwargs["config_str"])),
        )

        channels = checkpoint.get(
            "channels",
            experiment_info.get("channels", args_info.get("fixed_channels", model_kwargs["channels"])),
        )
        try:
            model_kwargs["channels"] = int(channels)
        except (TypeError, ValueError):
            pass

        for key in ("use_gpdc", "use_residual", "use_lmm", "use_sdpm", "use_deep_supervision"):
            model_kwargs[key] = bool(
                checkpoint.get(key, experiment_info.get(key, args_info.get(key, model_kwargs[key])))
            )

        return model_kwargs

    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        model = self.build_model(**self._infer_model_kwargs(checkpoint))

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model.eval()

        param_count = None
        lambda_ds = None
        seed = None
        if isinstance(checkpoint, dict):
            param_count = checkpoint.get("parameters", None)
            lambda_ds = checkpoint.get("lambda_ds", None)
            seed = checkpoint.get("seed", None)

        return model, param_count, lambda_ds, seed

    def calculate_metrics(self, pred, target, threshold=None):
        pred_prob = pred.astype(np.float32)
        target_binary = target.astype(np.float32)
        if threshold is None:
            threshold = self.threshold

        pred_binary = (pred_prob > threshold).astype(np.float32)
        pred_flat = pred_binary.reshape(-1)
        target_flat = target_binary.reshape(-1)
        pred_prob_flat = pred_prob.reshape(-1)

        try:
            tn, fp, fn, tp = confusion_matrix(target_flat, pred_flat, labels=[0, 1]).ravel()
        except ValueError:
            tp = float(((pred_flat == 1) & (target_flat == 1)).sum())
            fp = float(((pred_flat == 1) & (target_flat == 0)).sum())
            fn = float(((pred_flat == 0) & (target_flat == 1)).sum())
            tn = float(((pred_flat == 0) & (target_flat == 0)).sum())

        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

        try:
            auc = roc_auc_score(target_flat, pred_prob_flat)
        except ValueError:
            auc = float("nan")

        try:
            mcc = matthews_corrcoef(target_flat, pred_flat)
        except ValueError:
            mcc = 0.0

        f1 = 2 * precision * sensitivity / (precision + sensitivity + 1e-8)

        return {
            "Sensitivity": float(sensitivity),
            "Specificity": float(specificity),
            "Precision": float(precision),
            "Accuracy": float(accuracy),
            "F1": float(f1),
            "AUC": float(auc),
            "MCC": float(mcc),
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "Threshold": float(threshold),
        }

    def search_best_threshold(self, pred, target):
        base_metrics = self.calculate_metrics(pred, target, threshold=self.threshold)
        f1_base = base_metrics["F1"]

        best_metrics = base_metrics
        best_thr = self.threshold
        thr = self.thr_min
        while thr <= self.thr_max + 1e-8:
            metrics_t = self.calculate_metrics(pred, target, threshold=thr)
            if metrics_t["F1"] >= f1_base - self.f1_drop_tolerance:
                if metrics_t["Sensitivity"] > best_metrics["Sensitivity"] + 1e-6:
                    best_metrics = metrics_t
                    best_thr = thr
            thr += self.thr_step

        return best_metrics, best_thr, f1_base

    def evaluate_single_model(self, model, test_loader, lambda_value, seed):
        print(f"Evaluating model: lambda={lambda_value:.2f}, seed={seed}")
        wrapped_model = WrappedDeepSupervisionModel(model)

        all_predictions = []
        all_targets = []
        infer_total_s = 0.0
        infer_count = 0

        with torch.no_grad():
            for images, masks, _img_names in tqdm(
                test_loader,
                desc=f"Evaluating lambda={lambda_value:.2f}, seed={seed}",
            ):
                for i in range(images.size(0)):
                    image = images[i:i + 1]
                    mask = masks[i:i + 1]

                    if self.device.type == "cuda":
                        torch.cuda.synchronize()
                    t0 = datetime.now()
                    prediction = predict_full_image(
                        model=wrapped_model,
                        image=image,
                        patch_size=48,
                        stride=16,
                        device=self.device,
                    )
                    if self.device.type == "cuda":
                        torch.cuda.synchronize()

                    infer_total_s += (datetime.now() - t0).total_seconds()
                    infer_count += 1
                    all_predictions.append(prediction.cpu().numpy())
                    all_targets.append(mask.cpu().numpy())

        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        if infer_count > 0 and infer_total_s > 0:
            time_ms = infer_total_s * 1000.0 / infer_count
            fps = infer_count / infer_total_s
        else:
            time_ms = float("nan")
            fps = float("nan")

        if not self.optimize_threshold:
            metrics = self.calculate_metrics(all_predictions, all_targets)
            metrics["FPS"] = float(fps)
            metrics["Time(ms)"] = float(time_ms)
            print(f"  Fixed threshold: {self.threshold:.2f}")
            print(f"  F1: {metrics['F1']:.4f}")
            print(f"  AUC: {metrics['AUC']:.4f}")
            return metrics

        best_metrics, best_thr, f1_base = self.search_best_threshold(all_predictions, all_targets)
        best_metrics["F1_base_0_5"] = float(f1_base)
        best_metrics["BestThreshold"] = float(best_thr)
        best_metrics["FPS"] = float(fps)
        best_metrics["Time(ms)"] = float(time_ms)
        print(f"  Base F1 @ threshold={self.threshold:.2f}: {f1_base:.4f}")
        print(f"  Best threshold: {best_thr:.2f}")
        return best_metrics


def evaluate_lambda_grid(models_dir, data_dir, device, args):
    if args.lambda_list is None:
        lambda_list = [round(0.1 * i, 1) for i in range(10)]
    else:
        lambda_list = parse_float_list(args.lambda_list)

    if args.seeds is None:
        seeds = [20, 42, 80]
    else:
        seeds = parse_int_list(args.seeds)

    print(f"Lambda list: {lambda_list}")
    print(f"Seed list: {seeds}")

    test_loader = get_test_loader(
        data_dir,
        batch_size=1,
        num_workers=args.num_workers,
    )

    evaluator = LambdaGridEvaluator(
        device=device,
        fixed_config=args.fixed_config,
        fixed_channels=args.fixed_channels,
        threshold=args.threshold,
        optimize_threshold=args.optimize_threshold,
        thr_min=args.thr_min,
        thr_max=args.thr_max,
        thr_step=args.thr_step,
        f1_drop_tolerance=args.f1_drop_tolerance,
    )

    rows = []
    for lambda_value in lambda_list:
        lambda_str = f"{lambda_value:.1f}".replace(".", "_")
        lambda_dir = f"lambda_{lambda_str}"

        for seed in seeds:
            seed_dir = f"seed{seed}"
            model_path = os.path.join(models_dir, lambda_dir, seed_dir, "best_model.pth")

            print("\n---------------------------------------------")
            print(f"Check model: lambda={lambda_value:.2f}, seed={seed}")
            print(f"Model path: {model_path}")

            if not os.path.exists(model_path):
                print("  -> model not found, skip")
                continue

            try:
                model, param_count, lambda_ckpt, seed_ckpt = evaluator.load_model(model_path)
            except Exception as exc:
                print(f"  -> failed to load model: {exc}")
                continue

            param_num = int(param_count) if param_count is not None else 0
            params_m = float(param_num) / 1e6
            flops_g = estimate_flops_g(
                WrappedDeepSupervisionModel(model),
                input_size=(1, 1, 48, 48),
                device=device,
            )
            metrics = evaluator.evaluate_single_model(model, test_loader, lambda_value, seed)

            row = {
                "lambda": float(lambda_value),
                "seed": int(seed),
                "Parameters": param_num,
                "lambda_ckpt": float(lambda_ckpt) if lambda_ckpt is not None else lambda_value,
                "seed_ckpt": int(seed_ckpt) if seed_ckpt is not None else int(seed),
            }
            fps = float(metrics.pop("FPS", float("nan")))
            time_ms = float(metrics.pop("Time(ms)", float("nan")))
            row.update(metrics)
            rows.append(row)

            metric_row = {
                "Accuracy": row.get("Accuracy", float("nan")),
                "Sensitivity": row.get("Sensitivity", float("nan")),
                "Specificity": row.get("Specificity", float("nan")),
                "F1": row.get("F1", float("nan")),
                "AUC": row.get("AUC", float("nan")),
                "Params(M)": float(params_m),
                "FLOPs(G)": float(flops_g),
                "FPS": float(fps),
                "Time(ms)": float(time_ms),
            }
            print_terminal_metrics(metric_row)

    return rows


def summarize_lambda_results(rows, save_dir):
    if not rows:
        print("No results. rows is empty.")
        return None, None

    df = pd.DataFrame(rows)
    raw_path = os.path.join(save_dir, "lambda_grid_raw_results.csv")
    df.to_csv(raw_path, index=False, float_format="%.6f")
    print(f"Saved raw results: {raw_path}")

    metrics_to_agg = ["F1", "AUC", "Accuracy", "Sensitivity", "Specificity", "Precision", "MCC"]
    agg_dict = {metric: ["mean", "std"] for metric in metrics_to_agg if metric in df.columns}

    grouped = df.groupby("lambda").agg(agg_dict)
    grouped.columns = [f"{col[0]}_{col[1]}" for col in grouped.columns.values]
    grouped = grouped.reset_index().sort_values("lambda")

    summary_path = os.path.join(save_dir, "lambda_grid_summary.csv")
    grouped.to_csv(summary_path, index=False, float_format="%.6f")
    print(f"Saved summary results: {summary_path}")

    return df, grouped


def plot_lambda_f1_curve(summary_df, save_dir):
    if summary_df is None or summary_df.empty or "F1_mean" not in summary_df.columns:
        print("summary_df is empty or missing F1_mean, skip plot.")
        return

    lambdas = summary_df["lambda"].values
    f1_mean = summary_df["F1_mean"].values
    f1_std = summary_df["F1_std"].values if "F1_std" in summary_df.columns else None

    fig, ax = plt.subplots(figsize=(6, 4))
    if f1_std is not None:
        ax.errorbar(lambdas, f1_mean, yerr=f1_std, marker="o", linestyle="-")
    else:
        ax.plot(lambdas, f1_mean, marker="o", linestyle="-")

    ax.set_xlabel("lambda (deep supervision strength)")
    ax.set_ylabel("F1 (mean over seeds)")
    ax.set_title("DRIVE: lambda grid")
    ax.grid(True)

    best_idx = int(np.argmax(f1_mean))
    lambda_star = float(lambdas[best_idx])
    f1_star = float(f1_mean[best_idx])
    ax.scatter([lambda_star], [f1_star])
    ax.annotate(
        f"lambda*={lambda_star:.1f}\nF1={f1_star:.4f}",
        xy=(lambda_star, f1_star),
        xytext=(lambda_star + 0.05, f1_star + 0.0005),
        arrowprops=dict(arrowstyle="->"),
        ha="left",
        va="bottom",
    )

    fig.tight_layout()
    save_path = os.path.join(save_dir, "lambda_grid_f1_curve.png")
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved F1 curve: {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DRIVE lambda grid evaluation")

    parser.add_argument("--models_dir", type=str, default="./models/drive", help="Checkpoint root.")
    parser.add_argument("--data_dir", type=str, default="./data/DRIVE", help="Dataset root.")
    parser.add_argument("--results_dir", type=str, default="./results/drive", help="Result directory.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu.")
    parser.add_argument("--num_workers", type=int, default=4, help="Data loader workers.")

    parser.add_argument("--fixed_config", type=str, default="C-[V]脳11", help="Backbone config string.")
    parser.add_argument("--fixed_channels", type=int, default=32, help="Backbone channels.")

    parser.add_argument("--lambda_list", type=str, default=None, help="Comma-separated lambda list.")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seed list.")

    parser.add_argument("--threshold", type=float, default=0.40, help="Fixed binarization threshold.")
    parser.add_argument("--optimize_threshold", action="store_true", help="Search threshold per checkpoint.")
    parser.add_argument("--thr_min", type=float, default=0.30, help="Threshold search min.")
    parser.add_argument("--thr_max", type=float, default=0.60, help="Threshold search max.")
    parser.add_argument("--thr_step", type=float, default=0.01, help="Threshold search step.")
    parser.add_argument(
        "--f1_drop_tolerance",
        type=float,
        default=0.0005,
        help="Maximum allowed F1 drop when preferring higher sensitivity.",
    )

    args = parser.parse_args()
    try:
        args.fixed_config = normalize_config_string(args.fixed_config)
    except ValueError:
        args.fixed_config = "C-[V]x11"
    apply_model_path_aliases(args, ("models_dir",))
    os.makedirs(args.results_dir, exist_ok=True)

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    print(f"Models dir: {args.models_dir}")
    print(f"Data dir: {args.data_dir}")
    print(f"Results dir: {args.results_dir}")
    print(f"Optimize threshold: {args.optimize_threshold}")

    if not os.path.exists(args.models_dir):
        print(f"Error: models directory does not exist: {args.models_dir}")
        return

    start_time = datetime.now()
    rows = evaluate_lambda_grid(args.models_dir, args.data_dir, device, args)
    if not rows:
        print("No evaluable (lambda, seed) pairs were found.")
        return

    df_raw, df_summary = summarize_lambda_results(rows, args.results_dir)

    raw_json_path = os.path.join(args.results_dir, "lambda_grid_raw_results.json")
    with open(raw_json_path, "w", encoding="utf-8") as f:
        json.dump(safe_convert_for_json(df_raw.to_dict(orient="records")), f, indent=2, ensure_ascii=False)
    print(f"Saved raw JSON: {raw_json_path}")

    summary_json_path = os.path.join(args.results_dir, "lambda_grid_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(safe_convert_for_json(df_summary.to_dict(orient="records")), f, indent=2, ensure_ascii=False)
    print(f"Saved summary JSON: {summary_json_path}")

    lambda_star_info = {}
    if df_summary is not None and not df_summary.empty and "F1_mean" in df_summary.columns:
        best_idx = df_summary["F1_mean"].idxmax()
        lambda_star = float(df_summary.loc[best_idx, "lambda"])
        f1_star = float(df_summary.loc[best_idx, "F1_mean"])
        lambda_star_info = {
            "lambda_star": lambda_star,
            "F1_mean": f1_star,
        }
        print(f"Best lambda by F1_mean: {lambda_star:.2f} (F1_mean={f1_star:.4f})")

    star_path = os.path.join(args.results_dir, "lambda_star_summary.json")
    with open(star_path, "w", encoding="utf-8") as f:
        json.dump(safe_convert_for_json(lambda_star_info), f, indent=2, ensure_ascii=False)
    print(f"Saved lambda* summary: {star_path}")

    plot_lambda_f1_curve(df_summary, args.results_dir)

    duration = datetime.now() - start_time
    print("\n" + "=" * 60)
    print("DRIVE evaluation finished")
    print(f"Total duration: {duration}")
    print(f"Results saved in: {args.results_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
