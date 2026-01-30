from codecarbon import EmissionsTracker
import subprocess
import sys
import os

if __name__ == "__main__":
    logs_dir = os.path.join(os.getcwd(), "logs/codecarbon")
    os.makedirs(logs_dir, exist_ok=True)

    tracker = EmissionsTracker(
        project_name="FairDICE",
        output_dir=logs_dir,
        output_file="fairdice_runs_.csv",
        save_to_file=True,
    )

    tracker.start()

    cmd = ["python", "main.py"] + sys.argv[1:]

    try:
        return_code = subprocess.call(cmd)
    finally:
        tracker.stop()

    sys.exit(return_code)