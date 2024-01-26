import argparse
from birdnetlib import RecordingBuffer
from birdnetlib.analyzer import Analyzer
from datetime import datetime
from pathlib import Path
import soundfile as sf
import sounddevice as sd
from dataclasses import dataclass
from pathlib import Path
import wandb


@dataclass
class Args:
    file: Path | None


def analyze_and_publish_detections(analyzer, data, samplerate):
    print("analyzing audio")
    recording = RecordingBuffer(
        analyzer,
        buffer=data,
        rate=samplerate,
        # lat=35.4244,
        # lon=-120.7463,
        date=datetime.now(),
        min_conf=wandb.config["min_conf"],
    )
    recording.analyze()

    print("printing detections")
    detections_table = wandb.Table(["audio", "label", "confidence"])
    for i, d in enumerate(recording.detections):
        print("detection", i, d)
        start = int(samplerate * d["start_time"])
        end = int(samplerate * d["end_time"])
        segment = data[start:end]
        detections_table.add_data(
            wandb.Audio(segment, samplerate), d["label"], d["confidence"]
        )

    wandb.log({"detections": detections_table})


def run(args: Args):
    wandb.init(
        # set the wandb project where this run will be logged
        project="bird-detection",
        # track hyperparameters and run metadata
        config={
            "min_conf": 0.25,
            # "architecture": "CNN",
            # "dataset": "CIFAR-100",
            # "epochs": 10,
        },
    )

    print("loading analyzer model")
    analyzer = Analyzer()

    # perform oneshot analysis for file input
    if args.file:
        print("reading audio from file", args.file)
        data, samplerate = sf.read(args.file)
        analyze_and_publish_detections(analyzer, data, samplerate)
        return

    while True:
        print("recording 60s of audio from default mic")
        duration = 60.0  # seconds
        samplerate = 48000
        data = sd.rec(
            int(duration * samplerate),
            samplerate=samplerate,
            channels=1,
            blocking=True,
        )[:, 0]
        sd.wait()
        analyze_and_publish_detections(analyzer, data, samplerate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--file", type=Path, help="read input file instead default microphone"
    )
    args = parser.parse_args()

    run(Args(file=args.file))
