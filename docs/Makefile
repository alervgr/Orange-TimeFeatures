SPHINXBUILD ?= python -m sphinx
SOURCEDIR = .
BUILDDIR = build

.PHONY: html clean

html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html"

clean:
	rm -rf "$(BUILDDIR)"
