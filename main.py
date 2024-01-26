import argparse
from birdnetlib import RecordingBuffer
from birdnetlib.analyzer import Analyzer
from datetime import datetime
from pathlib import Path
import soundfile as sf
import soundcard as sc
from dataclasses import dataclass
from pathlib import Path
import wandb


@dataclass
class Args:
    file: Path | None


def run(args: Args):
    wandb.init(
        # set the wandb project where this run will be logged
        project="bird-detection",
        # track hyperparameters and run metadata
        config={
            "architecture": "CNN",
            "dataset": "CIFAR-100",
            "epochs": 10,
        },
    )

    print("loading analyzer model")
    analyzer = Analyzer()

    if args.file:
        print("reading audio from file", args.file)
        data, samplerate = sf.read(args.file)
    else:
        print("recording 30s of audio from default mic")
        mic = sc.default_microphone()
        samplerate = 48000
        data = mic.record(samplerate=samplerate, numframes=30 * samplerate)

    print("analyzing audio")
    recording = RecordingBuffer(
        analyzer,
        buffer=data,
        rate=samplerate,
        # lat=35.4244,
        # lon=-120.7463,
        date=datetime.now(),
        min_conf=0.25,
    )
    recording.analyze()

    print("printing detections")
    for i, d in enumerate(recording.detections):
        print("detection", i, d)
        # {'common_name': 'American Goldfinch', 'scientific_name': 'Spinus tristis', 'start_time': 3.0, 'end_time': 6.0, 'confidence': 0.647074818611145, 'label': 'Spinus tristis_American Goldfinch'}
        start = int(samplerate * d["start_time"])
        end = int(samplerate * d["end_time"])
        segment = data[start:end]
        # name
        # outpath = Path(
        #     "outputs",
        #     path.name,
        #     f"{d['label']}_{d['start_time']}_{d['end_time']}_{d['confidence']}.ogg",
        # )
        # outpath.parent.mkdir(exist_ok=True, parents=True)
        # print("saving segment to", outpath)
        # sf.write(str(outpath), segment, samplerate, format="ogg", subtype="vorbis")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--file", type=Path, help="read input file instead default microphone"
    )
    args = parser.parse_args()

    run(Args(file=args.file))
