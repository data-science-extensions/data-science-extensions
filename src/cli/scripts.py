## --------------------------------------------------------------------------- #
##  Setup                                                                   ####
## --------------------------------------------------------------------------- #


# StdLib Imports
import subprocess


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
