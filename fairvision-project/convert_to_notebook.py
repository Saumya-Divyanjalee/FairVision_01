import json
import re

INPUT_FILE  = "notebooks/main_notebook.py"
OUTPUT_FILE = "notebooks/main.ipynb"

with open(INPUT_FILE, "r") as f:
    content = f.read()

# Split on ## SECTION markers (used as cell boundaries)
raw_sections = re.split(r"\n(?=# ={10,})", content)

cells = []

def make_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip()
    }

def make_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip()
    }

for section in raw_sections:
    section = section.strip()
    if not section:
        continue

    # Extract section header comment as markdown
    lines = section.split("\n")
    header_lines = []
    code_lines   = []
    in_header    = True

    for line in lines:
        if in_header and line.startswith("#"):
            header_lines.append(line.lstrip("# ").strip())
        else:
            in_header = False
            code_lines.append(line)

    if header_lines:
        md_text = "## " + " — ".join(h for h in header_lines if h)
        cells.append(make_markdown_cell(md_text))

    code_text = "\n".join(code_lines).strip()
    if code_text:
        cells.append(make_code_cell(code_text))

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": cells
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(notebook, f, indent=2)

print(f"Notebook saved to: {OUTPUT_FILE}")
print(f"Total cells created: {len(cells)}")
