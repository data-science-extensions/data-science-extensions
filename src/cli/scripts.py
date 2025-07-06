## --------------------------------------------------------------------------- #
##  Setup                                                                   ####
## --------------------------------------------------------------------------- #


# StdLib Imports
import re
import subprocess
import sys
from pathlib import Path
from typing import Literal, LiteralString


## --------------------------------------------------------------------------- #
##  Generic                                                                 ####
## --------------------------------------------------------------------------- #


def run_command(*command) -> None:
    print("\n", " ".join(command), sep="", flush=True)
    subprocess.run(command, check=True)


def uv_sync() -> None:
    run_command("uv", "sync", "--all-groups", "--native-tls", "--link-mode=copy")


def lint_check() -> None:
    lint()
    check()


def get_all_files(*suffixes) -> list[str]:
    return [
        str(p)
        for p in Path("./").glob("**/*")
        if ".venv" not in p.parts and not p.parts[0].startswith(".") and p.is_file() and p.suffix in {*suffixes}
    ]


## --------------------------------------------------------------------------- #
##  Linting                                                                 ####
## --------------------------------------------------------------------------- #


def run_black() -> None:
    run_command("black", "--config=pyproject.toml", "./")


def run_blacken_docs() -> None:
    run_command("blacken-docs", "--line-length=120", *get_all_files(".md", ".py", ".ipynb"))


def run_isort() -> None:
    run_command("isort", "--settings-file=pyproject.toml", "./")


def run_pycln() -> None:
    run_command("pycln", "--config=pyproject.toml", "src/")


def run_pyupgrade() -> None:
    run_command("pyupgrade", "--py3-plus", *get_all_files(".py"))


def lint() -> None:
    run_black()
    run_blacken_docs()
    run_isort()
    run_pycln()


## --------------------------------------------------------------------------- #
##  Checking                                                                ####
## --------------------------------------------------------------------------- #


def check_black() -> None:
    run_command("black", "--check", "--config=pyproject.toml", "./")


def check_blacken_docs() -> None:
    run_command("blacken-docs", "--check", "--line-length=120", *get_all_files(".md", ".py", ".ipynb"))


def check_mypy() -> None:
    run_command("mypy", "--install-types", "--non-interactive", "--config-file=pyproject.toml", "./src")


def check_isort() -> None:
    run_command("isort", "--check", "--settings-file=pyproject.toml", "./")


def check_codespell() -> None:
    run_command("codespell", "--toml=pyproject.toml", "src/", "*.py")


def check_pylint() -> None:
    run_command("pylint", "--rcfile=pyproject.toml", "src/")


def check_pycln() -> None:
    run_command("pycln", "--config=pyproject.toml", "src/")


def check_build() -> None:
    run_command("uv", "build", "--out-dir=dist")
    run_command("rm", "--recursive", "dist")


def check_mkdocs() -> None:
    run_command("mkdocs", "build", "--site-dir=temp")
    run_command("rm", "--recursive", "temp")


def check_pytest() -> None:
    run_command("pytest", "--config-file=pyproject.toml")


def check() -> None:
    check_black()
    check_blacken_docs()
    # check_mypy()
    check_isort()
    check_codespell()
    check_pylint()
    check_pycln()
    # check_mkdocs()
    # check_build()
    # check_pytest()


## --------------------------------------------------------------------------- #
##  Reformatting                                                            ####
## --------------------------------------------------------------------------- #


def extract_sections_from_markdown_file(
    file_path: str,
    section_name: Literal["pandas", "sql", "pyspark", "polars"],
) -> None:
    """
    Summary:
        Extracts a specific section from a markdown file and saves it to a new file.

    Details:
        This function reads a markdown file, searches for tabbed sections defined by `=== "Tab Name"`, and extracts the content of the specified tab section. It then writes the content to a new markdown file with the section name as a suffix.

    Args:
        file_path (str):
            Path to the markdown file to process.
        section_name (Literal["pandas", "sql", "pyspark", "polars"]):
            The name of the section to extract. This should match the tab name in the markdown file.

    Notes:
        - The function expects the markdown file to have sections defined with `=== "Tab Name"`.
        - It will create a new file with the same name as the original but with `-{section_name}` appended before the file extension.
        - If the specified section is not found, it will print a message and exit without creating a new file.
        - If the file does not exist or is not a markdown file, it will print an error message and exit.
        - The function uses regular expressions to find and extract the specified section.
        - It handles multiple consecutive newlines that may occur after removing sections.
        - The function is case-insensitive when matching the section name.
        - The output file will overwrite any existing file with the same name.
    """

    # Load
    file = Path(file_path)

    # Check exists
    if not file.exists():
        print(f"File {file_path} does not exist.")
        return

    # Check file extension
    if not file.suffix == ".md":
        print(f"File {file_path} is not a markdown file.")
        return

    # Read the entire file as a single string
    with open(file, "r", encoding="utf-8") as f:
        content: str = f.read()

    # Create regex pattern to match tab sections
    # Pattern explanation:
    # ^=== "([^"]+)"$   - matches tab header line with captured group for tab name
    # (.*?)             - non-greedy capture of tab content (including newlines)
    # (?=^===|^[^\s]|$) - lookahead for next tab, non-indented line, or end of file
    tab_pattern: re.Pattern[str] = re.compile(
        pattern=r'^=== "([^"]+)"\s*\n((?:^[ \t]+.*\n*)*)',
        flags=re.MULTILINE,
    )

    # Find all tab sections
    matches: list[re.Match[str]] = list(tab_pattern.finditer(content))

    # Filter to keep only the desired section (case-insensitive)
    target_section_name: LiteralString = section_name.lower()
    kept_sections: list[re.Match[str]] = []

    if not matches:
        print(f"No tab sections found in {file_path}")
        return

    # Build the new content by replacing tab sections
    result_content: str = content
    target_section_name = section_name.lower()

    # Process matches in reverse order to maintain string positions
    for match in reversed(matches):
        tab_name: str = match.group(1)
        full_match: str = match.group(0)

        if tab_name.lower() == target_section_name:
            # Keep this section - no replacement needed
            continue
        else:
            # Remove this section
            result_content = result_content.replace(full_match, "", 1)

    # Check if we found any matching sections
    kept_sections: list[re.Match[str]] = [match for match in matches if match.group(1).lower() == target_section_name]

    if not kept_sections:
        print(f"No sections found for '{section_name}' in {file_path}")
        return

    # Clean up any multiple consecutive newlines that might have been created
    result_content = re.sub(r"\n{4,}", "\n\n\n", result_content)

    # Create output file path with section name as suffix
    file_stem: str = file.stem
    file_suffix = file.suffix
    output_file: Path = file.with_name(f"{file_stem}-{section_name.lower()}{file_suffix}")

    # Write the result (overwrite if exists)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result_content)

    print(f"Extracted '{section_name}' sections written to: {output_file}")
    print(f"Kept {len(kept_sections)} section(s), removed {len(matches) - len(kept_sections)} section(s)")


def extract_sections_from_markdown_file_cli() -> None:
    if len(sys.argv) < 3:
        print("Usage: extract-sections-from-markdown-file <file_path> <section_name>")
        sys.exit(1)

    file_path: str = sys.argv[1]
    section_name: Literal["pandas", "sql", "pyspark", "polars"] = sys.argv[2]

    # Validate section name
    valid_sections: list[str] = ["pandas", "sql", "pyspark", "polars"]
    if section_name.lower() not in valid_sections:
        print(f"Invalid section name '{section_name}'. Valid options are: {', '.join(valid_sections)}")
        sys.exit(1)

    extract_sections_from_markdown_file(file_path, section_name)


def reformat_file(file_path: str) -> str | None:
    """
    Summary:
        Dedent code blocks in a markdown file and save to a new file with 'reformatted' in the name.

    Details:
        Finds all Python code blocks that start with ```py and removes 4 spaces of indentation from each line within the block.

    Args:
        file_path (str):
            Path to the markdown file to process

    Notes:
        This function:

        1. Loads the specified markdown file
        2. Creates a new output filename with "reformatted" added
        3. Processes each line, tracking whether we're inside a Python code block
        4. When inside a code block, removes 4 spaces of indentation
        5. Preserves the code block markers (py and )
        6. Writes the processed content to the new file

        The CLI entry point you already have will work correctly with this implementation.
    """

    # Load
    file = Path(file_path)

    # Check exists
    if not file.exists():
        print(f"File {file_path} does not exist.")
        return

    # Create output file path
    file_stem: str = file.stem  # Get filename without extension
    file_suffix: str = file.suffix  # Get file extension
    output_file: Path = file.with_name(f"{file_stem}-reformatted{file_suffix}")

    # Read the file
    with open(file, "r") as f:
        lines: list[str] = f.readlines()

    # Process the content
    in_code_block = False

    # Clean lines
    for index, line in enumerate(lines):

        line: str

        # Check for code block start
        if not in_code_block:

            # Dedent lines
            if line.startswith("    "):
                line = line[4:]

            # Handle code block start
            if line.startswith("```py"):
                in_code_block = True
                line = line.replace("```py", "```python")

            # Convert headings
            if line.startswith("==="):
                line = line.replace("=== ", "### ").replace('"', "")

            # Drop lines
            if "--8<--" in line:
                line = ""

        # Check for code block end
        elif in_code_block:

            # Handle code block end
            if line.strip().startswith("```"):
                in_code_block = False

            # Dedent code block content
            if line.startswith("    "):
                line = line[4:]

        lines[index] = line

    # Write the result
    with open(output_file, "w") as f:
        f.writelines(lines)

    print(f"Reformatted file written to: {output_file}")

    return str(output_file)


def reformat_file_cli() -> None:
    if len(sys.argv) < 2:
        print("Usage: reformat-file <file_path>")
        sys.exit(1)
    reformat_file(sys.argv[1])


def convert_markdown_to_notebook(input_file_path: str) -> str | None:
    """
    Summary:
        Converts a markdown file to a Jupyter notebook using jupytext and formats it with black.

    Details:
        This function checks if the input file exists and is a markdown file. If so, it uses jupytext to convert the markdown file to a Jupyter notebook format, applying black for formatting. The output notebook file will have the same name as the input file but with the `.ipynb` extension.

    Args:
        input_file_path (str):
            Path to the markdown file to convert.

    Returns:
        str | None:
            The path to the converted Jupyter notebook file, or None if the input file does not exist or is not a markdown file.

    Notes:
        - The function uses the `jupytext` command-line tool to perform the conversion.
        - It applies `black` formatting to the notebook after conversion.
        - If the input file does not exist or is not a markdown file, it prints an error message and returns None.
        - The output file will be created in the same directory as the input file.
        - The output file will overwrite any existing file with the same name.
    """

    if not Path(input_file_path).exists():
        print(f"Input file {input_file_path} does not exist.")
        return
    if not input_file_path.endswith(".md"):
        print(f"Input file {input_file_path} is not a markdown file.")
        return

    output_file = input_file_path.replace(".md", ".ipynb")

    run_command(
        "jupytext",
        "--to=notebook",
        "--update",
        "--pipe=black",
        input_file_path,
        f"--output={output_file}",
    )

    print(f"Converted {input_file_path} to {output_file}")

    return str(output_file)


def convert_markdown_to_notebook_cli() -> None:
    if len(sys.argv) < 2:
        print("Usage: convert-markdown-to-notebook <file_path>")
        sys.exit(1)
    convert_markdown_to_notebook(sys.argv[1])


def format_and_convert(file_path: str) -> None:
    reformatted_file: str = reformat_file(file_path)
    convert_markdown_to_notebook(reformatted_file)


def format_and_convert_cli() -> None:
    if len(sys.argv) < 2:
        print("Usage: format-and-convert <file_path>")
        sys.exit(1)
    format_and_convert(sys.argv[1])
