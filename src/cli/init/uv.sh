curl -LsSf https://astral.sh/uv/install.sh | sh
uv self update
uv init --python 3.13 --no-cache --name "data-science-extensions" --description "Data Science Extensions" --package
uv addn --link-mode=copy --no-cache typeguard
uv addn --link-mode=copy --no-cache --group "dev" black blacken-docs pre-commit isort codespell pyupgrade pylint pycln ipykernel
uv addn --link-mode=copy --no-cache --group "docs" mkdocs mkdocs-material mkdocstrings mkdocstrings-python mkdocs-coverage mkdocs-autorefs livereload mike black docstring-inheritance
uv syncn --link-mode=copy --no-cache --all-groups