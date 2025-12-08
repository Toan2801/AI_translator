# Translation CLI Tool

## Overview

This project provides a command-line translation tool built with Python
and OpenAI API.\
It supports multiple source languages and uses external `.txt` prompt
files for full customization.

Supported source language codes: - `auto` -- auto detect - `zh` --
Classical Chinese / Chinese - `hv` -- Hán--Việt phonetic - `th` --
Thai - `lo` -- Lao - `en` -- English - `fr` -- French - `other` -- other
languages

Translation modes: - `smooth` (default) - `literal`

------------------------------------------------------------------------

## Folder Structure

    project/
    │
    ├── translator_bot.py
    ├── example.txt
    ├── input.txt
    ├── output.txt
    ├── README.md
    └── prompts/
        ├── system_zh_smooth.txt
        ├── system_zh_literal.txt
        ├── assistant_zh_smooth.txt
        ├── assistant_zh_literal.txt
        ├── intro_zh.txt
        ├── ...

------------------------------------------------------------------------

## Installation

### 1. Create virtual environment (optional but recommended)

    python -m venv venv
    source venv/bin/activate   # macOS/Linux
    venv\Scripts\activate    # Windows

### 2. Install dependencies

    pip install -r requirements.txt

### 3. Create `.env` file

Create a file named `.env`:

    OPENAI_API_KEY=your_api_key_here

------------------------------------------------------------------------

## Usage

Run the tool:

    python translator_bot.py

You will be prompted to enter: 1. Input file path 2. Output file path 3.
Source language code

Example:

    Enter input file path (default: input.txt):
    Enter output file path (default: output.txt):
    Enter source language code [auto, zh, hv, en, th, lo, fr, other] (default: hv):

------------------------------------------------------------------------

## Prompt System

All AI instructions are stored in external `.txt` files inside the
`prompts/` directory.

File naming rules:

    system_{lang}_{mode}.txt
    assistant_{lang}_{mode}.txt
    intro_{lang}.txt

Examples: - `system_en_smooth.txt` - `assistant_th_literal.txt` -
`intro_fr.txt`

This allows full control over translation style without touching Python
code.

------------------------------------------------------------------------

## Notes

-   `example.txt` is optional and provides style guidance.
-   Large input files are automatically split into safe chunks.
-   Output is always written to the selected output file.

------------------------------------------------------------------------

## License

Private project -- free for internal and personal use.
