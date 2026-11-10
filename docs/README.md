# AlphaGenome docs

To build the docs locally from the repository root:

```shell
$ pip install -e ".[docs]"
$ mkdir -p docs/source/colabs && cp -r colabs/* docs/source/colabs/
$ sphinx-build -b html --keep-going docs/source docs/build/html
```

Then open `docs/build/html/index.html` in a browser.
