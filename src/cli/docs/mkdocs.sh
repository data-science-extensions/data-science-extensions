# docs-serve-static:
uv run --link-mode=copy mkdocs serve

# docs-serve-versioned:
uv run --link-mode=copy mike serve --branch=docs-site

# docs-build-static:
uv run --link-mode=copy mkdocs build --clean

# docs-build-versioned:
git config --global --list
git config --local --list
git remote -v
uv run --link-mode=copy mike --debug deploy --update-aliases --branch=docs-site --push $(VERSION) latest

# update-git-docs:
git add .
git commit -m "Build docs [skip ci]"
git push --force --no-verify --push-option ci.skip

# docs-check-versions:
uv run --link-mode=copy mike --debug list --branch=docs-site

# docs-delete-version:
uv run --link-mode=copy mike --debug delete --branch=docs-site $(VERSION)

# docs-set-default:
uv run --link-mode=copy mike --debug set-default --branch=docs-site --push latest

# build-static-docs:
docs-build-static
update-git-docs

# build-versioned-docs:
docs-build-versioned
docs-set-default
