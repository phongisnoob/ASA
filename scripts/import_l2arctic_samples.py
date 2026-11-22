
from pathlib import Path
from zipfile import ZipFile
import csv
import shutil
import sys

# Project root
ROOT = Path(__file__).resolve().parents[1]

ZIP_PATH = ROOT / "downloads" / "l2arctic" / "HQTV.zip"
AUDIO_DIR = ROOT / "examples" / "audio"
MANIFEST_PATH = ROOT / "examples" / "input_manifest.csv"

# Preserve the original synthetic demo files and manifest
AUDIO_BACKUP = ROOT / "examples" / "audio_demo_backup"
MANIFEST_BACKUP = ROOT / "examples" / "input_manifest_demo_backup.csv"

# Three recordings from the same Vietnamese L1 speaker
SAMPLES = [
    ("S001", "arctic_a0018", "ex_001.wav"),
    ("S002", "arctic_a0030", "ex_002.wav"),
    ("S003", "arctic_a0036", "ex_003.wav"),
]

def find_member(names, folder, filename):
    """Find a file in the speaker ZIP, regardless of root folder name."""
    suffix = f"{folder}/{filename}".lower()

    matches = [
        name for name in names
        if name.replace("\\", "/").lower().endswith(suffix)
    ]

    if not matches:
        raise FileNotFoundError(
            f"Cannot find {folder}/{filename} inside HQTV.zip"
        )

    return matches[0]


def main():
    if not ZIP_PATH.is_file():
        raise FileNotFoundError(
            f"HQTV.zip not found: {ZIP_PATH}\n"
            "Download it and put it in downloads/l2arctic/"
        )

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_BACKUP.mkdir(parents=True, exist_ok=True)

    with ZipFile(ZIP_PATH, "r") as archive:
        names = archive.namelist()

        # Check all six files before changing anything.
        selected = []

        for sample_id, stem, output_name in SAMPLES:
            wav_member = find_member(
                names, "wav", f"{stem}.wav"
            )
            txt_member = find_member(
                names, "transcript", f"{stem}.txt"
            )

            transcript = archive.read(txt_member).decode(
                "utf-8-sig"
            )
            transcript = " ".join(transcript.split())

            if not transcript:
                raise ValueError(
                    f"Empty transcript: {txt_member}"
                )

            selected.append({
                "sample_id": sample_id,
                "stem": stem,
                "output_name": output_name,
                "wav_member": wav_member,
                "txt_member": txt_member,
                "transcript": transcript,
            })

        # Back up the original demo manifest.
        if MANIFEST_PATH.exists() and not MANIFEST_BACKUP.exists():
            shutil.copy2(MANIFEST_PATH, MANIFEST_BACKUP)

        # Back up existing synthetic audio before replacing it.
        for item in selected:
            old_audio = AUDIO_DIR / item["output_name"]

            if old_audio.exists():
                backup_path = (
                    AUDIO_BACKUP /
                    (item["output_name"] + ".synthetic.bak")
                )
                if not backup_path.exists():
                    shutil.copy2(old_audio, backup_path)

        rows = []

        for item in selected:
            output_audio = AUDIO_DIR / item["output_name"]

            # Extract only the selected WAV.
            with archive.open(item["wav_member"]) as source:
                with output_audio.open("wb") as destination:
                    shutil.copyfileobj(source, destination)

            rows.append({
                "sample_id": item["sample_id"],
                "audio_path": f"audio/{item['output_name']}",
                "reference_text": item["transcript"],
                "speaker_id": "SPK_HQTV",
                "human_verbatim": "",
            })

            print(f"\nImported: {item['stem']}")
            print(f"  WAV: {output_audio}")
            print(f"  Transcript: {item['transcript']}")

        # Write a manifest whose transcripts match the downloaded WAVs.
        fieldnames = [
            "sample_id",
            "audio_path",
            "reference_text",
            "speaker_id",
            "human_verbatim",
        ]

        with MANIFEST_PATH.open(
            "w", newline="", encoding="utf-8-sig"
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=fieldnames
            )
            writer.writeheader()
            writer.writerows(rows)

    print("\nSuccessfully imported 3 L2-ARCTIC samples.")
    print(f"Audio directory: {AUDIO_DIR}")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Original demo backup: {AUDIO_BACKUP}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
