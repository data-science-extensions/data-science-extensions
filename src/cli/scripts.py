## --------------------------------------------------------------------------- #
##  Setup                                                                   ####
## --------------------------------------------------------------------------- #


# StdLib Imports
import subprocess
import sys
from pathlib import Path


## --------------------------------------------------------------------------- #
##  UV Processes                                                            ####
## --------------------------------------------------------------------------- #


def run_command(*command) -> None:
    print(" ".join(command))
    subprocess.run(command, check=True)


def uv_sync() -> None:
    run_command("uv", "sync", "--all-groups", "--native-tls", "--link-mode=copy")


## --------------------------------------------------------------------------- #
##  Linting                                                                 ####
## --------------------------------------------------------------------------- #


def run_black() -> None:
    run_command("black", "--config=pyproject.toml", "./")


def run_isort() -> None:
    run_command("isort", "--settings-file=pyproject.toml", "./")


def lint() -> None:
    run_black()
    run_isort()


## --------------------------------------------------------------------------- #
##  Checking                                                                ####
## --------------------------------------------------------------------------- #


def check_black() -> None:
    run_command("black", "--check", "--config=pyproject.toml", "./")


def check_mypy() -> None:
    run_command(
        "mypy",
        "--install-types",
        "--non-interactive",
        "--config-file=pyproject.toml",
        "./src",
    )


def check_isort() -> None:
    run_command("isort", "--check", "--settings-file=pyproject.toml", "./")


def check_codespell() -> None:
    run_command("codespell", "--toml", "pyproject.toml", "src/", "*.py")


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
    check_isort()
    # check_mypy()
    check_codespell()
    check_pylint()
    check_pycln()
    # check_mkdocs()
    # check_build()
    check_pytest()


## --------------------------------------------------------------------------- #
##  Reformatting                                                            ####
## --------------------------------------------------------------------------- #


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
    reformatted_file = reformat_file(file_path)
    assert reformatted_file is not None
    convert_markdown_to_notebook(reformatted_file)


def format_and_convert_cli() -> None:
    if len(sys.argv) < 2:
        print("Usage: format-and-convert <file_path>")
        sys.exit(1)
    format_and_convert(sys.argv[1])
