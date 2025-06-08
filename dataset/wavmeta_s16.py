import os
import argparse
import subprocess
import csv

def convert_wav_to_s16(meta_file, out_dir, identifier, col=1, concat=False):
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Converting WAVs listed in {meta_file} → {out_dir} (.s16)")
    s16_file_list = []

    with open(meta_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=",")  # auto-handle comma, tab, or space

        for line_no, row in enumerate(reader, 1):
            if not row or len(row) < col:
                continue

            wav_path = row[col - 1].strip()
            if not os.path.isfile(wav_path):
                print(f"Skipping line {line_no} (file not found): {wav_path}")
                continue

            abs_wav = os.path.abspath(wav_path)
            _, _, rel_path = abs_wav.partition(identifier)
            rel_path = identifier + rel_path
            dst_path = os.path.join(out_dir, os.path.splitext(rel_path)[0] + ".s16")
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)

            cmd = [
                "ffmpeg",
                "-loglevel", "error",
                "-y",
                "-i", abs_wav,
                "-ac", "1",
                "-ar", "16000",
                "-f", "s16le",
                dst_path
            ]

            try:
                subprocess.run(cmd, check=True)
                print(f"✓ {os.path.basename(wav_path)} → {dst_path}")
                s16_file_list.append(dst_path)
            except subprocess.CalledProcessError:
                print(f"✗ Failed: [{wav_path}]")

    print(f"Done. .s16 files are under {out_dir}")

    # Concatenate .s16 files into one output if flag is set
    if concat:
        concat_path = os.path.join(out_dir, "mergedinput.s16")
        with open(concat_path, "wb") as outfile:
            for s16_path in s16_file_list:
                with open(s16_path, "rb") as infile:
                    outfile.write(infile.read())
        print(f"All .s16 files concatenated to: {concat_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert WAVs from metadata list to 16kHz mono .s16")
    parser.add_argument("metadata_file", type=str, help="Path to metadata file")
    parser.add_argument("output_dir", type=str, help="Output directory for .s16 files")
    parser.add_argument("identifier", type=str, help="Base path for initial subfolder")
    parser.add_argument("--col", type=int, default=1, help="1-based column index for WAV paths (default: 1)")
    parser.add_argument("--concat", action="store_true", help="Concatenate all .s16 files into one")

    args = parser.parse_args()
    convert_wav_to_s16(args.metadata_file, args.output_dir, args.identifier, args.col, args.concat)
