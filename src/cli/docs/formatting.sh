uv run lint
uv run extract-sections-from-markdown-file docs/guides/querying-data/index.md pandas
uv run extract-sections-from-markdown-file docs/guides/querying-data/index.md sql
uv run extract-sections-from-markdown-file docs/guides/querying-data/index.md pyspark
uv run extract-sections-from-markdown-file docs/guides/querying-data/index.md polars
uv run reformat-and-convert-md-to-ipynb docs/guides/querying-data/index.md
uv run reformat-and-convert-md-to-ipynb docs/guides/querying-data/index-pandas.md
uv run reformat-and-convert-md-to-ipynb docs/guides/querying-data/index-sql.md
uv run reformat-and-convert-md-to-ipynb docs/guides/querying-data/index-pyspark.md
uv run reformat-and-convert-md-to-ipynb docs/guides/querying-data/index-polars.md
