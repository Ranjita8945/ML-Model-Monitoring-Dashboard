"""ML Model Monitoring Dashboard - Application Entrypoint.

Usage:
    python app.py                         # Launch the Streamlit dashboard
    python app.py --mode train            # Train the classification model
    python app.py --mode monitor          # Run a single monitoring cycle
    python app.py --mode pipeline         # Train -> predict -> monitor
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def launch_dashboard() -> None:
    """Start the Streamlit dashboard."""
    dashboard_path = PROJECT_ROOT / "dashboard" / "app.py"
    if not dashboard_path.exists():
        print(f"[ERROR] Dashboard entrypoint not found: {dashboard_path}")
        sys.exit(1)
    print("[INFO] Starting Streamlit dashboard...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)], check=True)


def run_train() -> None:
    """Train the classification model and persist artifacts."""
    from src.model.train import main as train_main

    train_main()


def run_monitor() -> None:
    """Run a single monitoring cycle (drift + performance + report)."""
    from src.monitoring.monitoring_report import run_monitoring_cycle

    run_monitoring_cycle()


def run_pipeline() -> None:
    """End-to-end: train -> predict -> monitor."""
    run_train()
    print("[INFO] Pipeline training complete. Monitoring cycle starting...")
    run_monitor()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ML Model Monitoring Dashboard"
    )
    parser.add_argument(
        "--mode",
        choices=["dashboard", "train", "monitor", "pipeline"],
        default="dashboard",
        help="Execution mode (default: dashboard)",
    )
    args = parser.parse_args()

    if args.mode == "dashboard":
        launch_dashboard()
    elif args.mode == "train":
        run_train()
    elif args.mode == "monitor":
        run_monitor()
    elif args.mode == "pipeline":
        run_pipeline()


if __name__ == "__main__":
    main()
