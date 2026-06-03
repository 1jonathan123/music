import argparse
import re
from pathlib import Path

import numpy as np
import sounddevice as sd


SAMPLE_RATE = 44100
DEFAULT_STEP_DURATION = 0.16
DEFAULT_BEATS_PER_BAR = 4
OUTPUT_TAIL_SECONDS = 0.05
VOLUME = 0.35

# Semitone positions relative to C
NOTE_OFFSETS = {
    "C": 0,
    "C#": 1,
    "DB": 1,
    "D": 2,
    "D#": 3,
    "EB": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "GB": 6,
    "G": 7,
    "G#": 8,
    "AB": 8,
    "A": 9,
    "A#": 10,
    "BB": 10,
    "B": 11,
}


def note_to_frequency(note_name, octave):
    """
    Convert a note like A4, C5, F#3, Bb4 into a frequency.
    A4 = 440 Hz.
    """
    note_name = note_name.upper()

    if note_name not in NOTE_OFFSETS:
        raise ValueError(f"Unknown note: {note_name}")

    # MIDI note formula:
    # C4 = MIDI 60
    # A4 = MIDI 69 = 440 Hz
    midi_number = 12 * (octave + 1) + NOTE_OFFSETS[note_name]
    return 440.0 * (2 ** ((midi_number - 69) / 12))


def envelope(length, attack=0.01, decay=0.08):
    env = np.ones(length)

    attack_len = int(attack * SAMPLE_RATE)
    decay_len = int(decay * SAMPLE_RATE)

    if attack_len > 0:
        env[:attack_len] = np.linspace(0, 1, attack_len)

    if decay_len > 0:
        env[-decay_len:] = np.linspace(1, 0, decay_len)

    return env


def make_note(freq, duration):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)

    wave = (
        np.sin(2 * np.pi * freq * t)
        + 0.35 * np.sin(2 * np.pi * freq * 2 * t)
        + 0.15 * np.sin(2 * np.pi * freq * 3 * t)
    )

    wave *= envelope(len(wave), attack=0.01, decay=0.06)
    return wave * VOLUME


def make_mute(duration):
    length = int(SAMPLE_RATE * duration)

    noise = np.random.uniform(-1, 1, length)
    decay = np.exp(-np.linspace(0, 12, length))

    t = np.linspace(0, duration, length, False)
    thump = np.sin(2 * np.pi * 90 * t) * np.exp(-np.linspace(0, 18, length))

    return (noise * decay * 0.6 + thump * 0.4) * VOLUME


def make_rest(duration):
    return np.zeros(int(SAMPLE_RATE * duration))


def tokenize_pattern(pattern, default_octave=4):
    """
    Turns a pattern string into tokens.

    Examples:
        X__XX_X_A4_XX_X__A4_B3__X_X_A4B4C5B4A4G4A4

    Supported:
        _       rest
        X       muted hit
        |       bar separator, only used with bpm timing
        A4      note with octave
        C#4     sharp note
        Bb3     flat note
        A       note using default octave
    """
    tokens = []
    i = 0

    while i < len(pattern):
        char = pattern[i]

        if char.isspace():
            i += 1
            continue

        if char == "|":
            tokens.append(("bar", None))
            i += 1
            continue

        if char == "_":
            tokens.append(("rest", None))
            i += 1
            continue

        if char.upper() == "X":
            tokens.append(("mute", None))
            i += 1
            continue

        if char.upper() in "ABCDEFG":
            note = char.upper()
            i += 1

            # Optional sharp/flat
            if i < len(pattern) and pattern[i] in ["#", "b"]:
                note += pattern[i].upper()
                i += 1

            # Optional octave number
            octave_match = re.match(r"\d+", pattern[i:])
            if octave_match:
                octave = int(octave_match.group())
                i += len(octave_match.group())
            else:
                octave = default_octave

            tokens.append(("note", (note, octave)))
            continue

        raise ValueError(f"Unknown symbol at position {i}: {char}")

    return tokens


def scan_pattern_parts(pattern):
    parts = []
    i = 0

    while i < len(pattern):
        char = pattern[i]

        if char.isspace():
            i += 1
            continue

        if char in ["_", "|"] or char.upper() == "X":
            parts.append(char)
            i += 1
            continue

        if char.upper() in "ABCDEFG":
            start = i
            i += 1

            if i < len(pattern) and pattern[i] in ["#", "b"]:
                i += 1

            octave_match = re.match(r"\d+", pattern[i:])
            if octave_match:
                i += len(octave_match.group())

            parts.append(pattern[start:i])
            continue

        raise ValueError(f"Unknown symbol at position {i}: {char}")

    return parts


def shape_pattern(pattern, steps_per_group, groups_per_line):
    if steps_per_group <= 0 or groups_per_line <= 0:
        raise ValueError("auto-shape values must be greater than zero.")

    return shape_pattern_with_line_counts(pattern, steps_per_group, [groups_per_line])


def shape_pattern_line(pattern_line, steps_per_group):
    if steps_per_group <= 0:
        raise ValueError("auto-shape values must be greater than zero.")

    groups = []
    group = []

    for part in scan_pattern_parts(pattern_line):
        if part == "|":
            if group:
                groups.append("".join(group))
                group = []
            groups.append("|")
            continue

        group.append(part)

        if len(group) == steps_per_group:
            groups.append("".join(group))
            group = []

    if group:
        groups.append("".join(group))

    return " ".join(groups)


def shape_pattern_with_line_counts(pattern, steps_per_group, line_group_counts):
    if steps_per_group <= 0:
        raise ValueError("auto-shape values must be greater than zero.")

    if not line_group_counts or any(count <= 0 for count in line_group_counts):
        raise ValueError("auto-shape line group counts must be greater than zero.")

    lines = []
    line_parts = []
    group = []
    line_groups = 0
    line_index = 0

    def current_groups_per_line():
        index = min(line_index, len(line_group_counts) - 1)
        return line_group_counts[index]

    def flush_line():
        nonlocal line_groups, line_index

        if line_parts:
            lines.append(" ".join(line_parts))
            line_parts.clear()
            line_groups = 0
            line_index += 1

    def flush_group():
        nonlocal line_groups

        if group:
            line_parts.append("".join(group))
            group.clear()
            line_groups += 1

            if line_groups == current_groups_per_line():
                flush_line()

    for part in scan_pattern_parts(pattern):
        if part == "|":
            flush_group()
            line_parts.append("|")
            continue

        group.append(part)

        if len(group) == steps_per_group:
            flush_group()

    flush_group()
    flush_line()

    return "\n".join(lines)


def is_setting_line(line):
    if "=" not in line:
        return False

    key, _ = line.split("=", 1)
    return key.strip().lower() in {"step_duration", "bpm", "default_octave", "pattern"}


def find_pattern_line_index(lines):
    for index, line in enumerate(lines):
        if is_setting_line(line) and line.split("=", 1)[0].strip().lower() == "pattern":
            return index

    return None


def pattern_body_lines(original_lines, pattern_line_index):
    if pattern_line_index is None:
        return [
            line.strip()
            for line in original_lines
            if line.strip() and not line.strip().startswith("#")
        ]

    lines = []
    pattern_value = original_lines[pattern_line_index].split("=", 1)[1].strip()
    if pattern_value:
        lines.append(pattern_value)

    index = pattern_line_index + 1
    while index < len(original_lines):
        line = original_lines[index]
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and is_setting_line(line):
            break
        if stripped and not stripped.startswith("#"):
            lines.append(stripped)
        index += 1

    return lines


def write_shaped_pattern_file(path, steps_per_group, groups_per_line=None):
    riff_path = Path(path)
    text = riff_path.read_text(encoding="utf-8")
    newline = "\r\n" if "\r\n" in text else "\n"
    original_lines = text.splitlines()

    pattern_line_index = find_pattern_line_index(original_lines)
    if groups_per_line is None:
        shaped_lines = [
            shape_pattern_line(line, steps_per_group)
            for line in pattern_body_lines(original_lines, pattern_line_index)
        ]
    else:
        pattern, _, _, _ = read_pattern_file(riff_path)
        shaped_pattern = shape_pattern(pattern, steps_per_group, groups_per_line)
        shaped_lines = shaped_pattern.splitlines()

    if pattern_line_index is None:
        new_lines = shaped_lines
    else:
        pattern_line = original_lines[pattern_line_index]
        indent = pattern_line[: len(pattern_line) - len(pattern_line.lstrip())]
        replacement = [f"{indent}pattern ="]
        replacement.extend(shaped_lines)

        suffix_index = pattern_line_index + 1
        while suffix_index < len(original_lines):
            line = original_lines[suffix_index]
            if line.strip() and not line.strip().startswith("#") and is_setting_line(line):
                break
            suffix_index += 1

        new_lines = (
            original_lines[:pattern_line_index]
            + replacement
            + original_lines[suffix_index:]
        )

    riff_path.write_text(newline.join(new_lines) + newline, encoding="utf-8")


def token_to_audio(token_type, value, duration):
    if token_type == "rest":
        return make_rest(duration)

    if token_type == "mute":
        return make_mute(duration)

    if token_type == "note":
        note_name, octave = value
        freq = note_to_frequency(note_name, octave)
        return make_note(freq, duration)

    raise ValueError(f"Cannot turn token into audio: {token_type}")


def split_tokens_into_bars(tokens):
    bars = [[]]

    for token_type, value in tokens:
        if token_type == "bar":
            if bars[-1]:
                bars.append([])
            continue

        bars[-1].append((token_type, value))

    return [bar for bar in bars if bar]


def pattern_to_audio(
    pattern,
    step_duration=DEFAULT_STEP_DURATION,
    default_octave=4,
    bpm=None,
    beats_per_bar=DEFAULT_BEATS_PER_BAR,
):
    chunks = []
    tokens = tokenize_pattern(pattern, default_octave=default_octave)

    if bpm is not None:
        if bpm <= 0:
            raise ValueError("bpm must be greater than zero.")

        bar_duration = beats_per_bar * 60.0 / bpm

        for bar in split_tokens_into_bars(tokens):
            token_duration = bar_duration / len(bar)
            for token_type, value in bar:
                chunks.append(token_to_audio(token_type, value, token_duration))
    else:
        for token_type, value in tokens:
            if token_type == "bar":
                raise ValueError(
                    "Bars require bpm timing; use bpm instead of step_duration."
                )

            chunks.append(token_to_audio(token_type, value, step_duration))

    if not chunks:
        return np.array([], dtype=np.float32)

    audio = np.concatenate(chunks)
    audio = np.concatenate([audio, make_rest(OUTPUT_TAIL_SECONDS)])

    # Prevent clipping
    max_amp = np.max(np.abs(audio))
    if max_amp > 0:
        audio = audio / max_amp * 0.9

    return audio


def play_pattern(
    pattern,
    step_duration=DEFAULT_STEP_DURATION,
    default_octave=4,
    bpm=None,
):
    audio = pattern_to_audio(
        pattern,
        step_duration=step_duration,
        default_octave=default_octave,
        bpm=bpm,
    )

    sd.play(audio, SAMPLE_RATE)
    sd.wait()


def read_pattern_file(path):
    """
    Reads a pattern file.

    File can be simple:

        X__XX_X_A4_XX_X__A4_B3__X_X_A4B4C5B4A4G4A4

    Or include settings:

        step_duration = 0.12
        default_octave = 4
        pattern = X__XX_X_A4_XX_X__A4_B3__X_X_A4B4C5B4A4G4A4

    Or use bpm timing, where each bar fills one 4-beat measure:

        bpm = 120
        default_octave = 4
        pattern =
            X__XX_X_ | A4_B3__
    """
    text = Path(path).read_text(encoding="utf-8")

    step_duration = None
    bpm = None
    default_octave = None
    pattern_lines = []

    for line in text.splitlines():
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        if "=" in line:
            key, value = line.split("=", 1)
            key = key.strip().lower()
            value = value.strip()

            if key == "step_duration":
                step_duration = float(value)
            elif key == "bpm":
                bpm = float(value)
            elif key == "default_octave":
                default_octave = int(value)
            elif key == "pattern":
                if value:
                    pattern_lines.append(value)
            else:
                raise ValueError(f"Unknown setting in file: {key}")
        else:
            pattern_lines.append(line)

    pattern = "".join(pattern_lines)

    if step_duration is not None and bpm is not None:
        raise ValueError("Use either step_duration or bpm in a riff file, not both.")

    return pattern, step_duration, bpm, default_octave


def main():
    parser = argparse.ArgumentParser(
        description="Turn a text pattern into simple music."
    )

    parser.add_argument(
        "file",
        nargs="?",
        help="Optional pattern file to run.",
    )

    parser.add_argument(
        "-p",
        "--pattern",
        help="Pattern string to play directly.",
    )

    parser.add_argument(
        "-s",
        "--step-duration",
        type=float,
        default=None,
        help="Duration of each symbol in seconds.",
    )

    parser.add_argument(
        "-b",
        "--bpm",
        type=float,
        default=None,
        help="Tempo for bar-based timing. Mutually exclusive with step_duration.",
    )

    parser.add_argument(
        "-o",
        "--default-octave",
        type=int,
        default=None,
        help="Octave used when a note has no octave number.",
    )

    parser.add_argument(
        "--auto-shape",
        nargs="+",
        type=int,
        metavar="N",
        help=(
            "Rewrite the riff file with groups of STEPS. Optional GROUPS sets "
            "groups per line; omitted preserves the current groups per line."
        ),
    )

    args = parser.parse_args()

    if args.auto_shape:
        if not args.file:
            raise SystemExit("--auto-shape requires a riff file.")
        if len(args.auto_shape) not in {1, 2}:
            raise SystemExit("--auto-shape expects STEPS or STEPS GROUPS.")

        steps_per_group = args.auto_shape[0]
        groups_per_line = args.auto_shape[1] if len(args.auto_shape) == 2 else None
        write_shaped_pattern_file(args.file, steps_per_group, groups_per_line)
        return

    file_pattern = None
    file_step_duration = None
    file_bpm = None
    file_default_octave = None

    if args.file:
        file_pattern, file_step_duration, file_bpm, file_default_octave = (
            read_pattern_file(args.file)
        )

    pattern = args.pattern or file_pattern
    if not pattern:
        raise SystemExit("Provide either a file or --pattern.")

    if args.step_duration is not None and args.bpm is not None:
        raise SystemExit("Use either --step-duration or --bpm, not both.")

    bpm = args.bpm if args.bpm is not None else file_bpm

    if args.step_duration is not None:
        if file_bpm is not None:
            raise SystemExit("Use either step_duration or bpm timing, not both.")
        step_duration = args.step_duration
    elif bpm is not None:
        step_duration = None
    else:
        step_duration = (
            file_step_duration
            if file_step_duration is not None
            else DEFAULT_STEP_DURATION
        )

    default_octave = (
        args.default_octave
        if args.default_octave is not None
        else file_default_octave
        if file_default_octave is not None
        else 4
    )

    play_pattern(
        pattern,
        step_duration=step_duration,
        default_octave=default_octave,
        bpm=bpm,
    )


if __name__ == "__main__":
    main()
