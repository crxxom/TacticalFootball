import argparse
import os
import subprocess
import sys

def run(cmd, cwd=None):
    print(f"Running: {cmd}")
    subprocess.check_call(cmd, shell=True, cwd=cwd)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="", help="Git repo URL")
    parser.add_argument("--project-dir", default="/content/RL_tacticalfootball")
    parser.add_argument("--run-train", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    project_dir = args.project_dir

    if args.repo:
        if not os.path.exists(project_dir):
            run(f"git clone {args.repo} {project_dir}")
        else:
            print("Project directory already exists.")
    else:
        if not os.path.exists(project_dir):
            raise FileNotFoundError(f"{project_dir} not found. Upload your folder or provide --repo.")

    run(f"{sys.executable} -m pip install --upgrade pip", cwd=project_dir)
    run(f"{sys.executable} -m pip install -r requirements.txt", cwd=project_dir)

    if args.skip_train:
        print("Skipping training.")
    else:
        if args.run_train:
            run(f"{sys.executable} train.py", cwd=project_dir)

    checkpoints_zip = os.path.join(project_dir, "checkpoints.zip")
    frames_zip = os.path.join(project_dir, "frames.zip")

    run("zip -r checkpoints.zip runs/checkpoints || true", cwd=project_dir)
    run("zip -r frames.zip runs/frames || true", cwd=project_dir)

    print("Artifacts created:")
    if os.path.exists(checkpoints_zip):
        print(f"- {checkpoints_zip}")
    if os.path.exists(frames_zip):
        print(f"- {frames_zip}")

if __name__ == "__main__":
    main()
