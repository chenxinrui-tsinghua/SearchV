"""Parse layer skip config files; vision prefix length shared with decoder (see llava_llama)."""
from typing import List, Tuple

# Decoder text region starts after this many prefix tokens (single-image LLaVA layout).
VISUAL_START_IDX = 35


def load_skip_config(file_path: str) -> List[Tuple[str, str]]:
    layer_drop_config: List[Tuple[str, str]] = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                cleaned_line = line.strip()

                if cleaned_line:
                    parts = cleaned_line.split()
                    if len(parts) == 2:
                        layer_drop_config.append((parts[0], parts[1]))
                    else:
                        print(f"Warning: Skipping malformed line: {line.strip()}")
                        continue

    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return layer_drop_config
