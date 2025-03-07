# Makefile for building the Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build  # sphinx-6.0+
SPHINXAPIDOC  = sphinx-apidoc # sphinx-6.0+
SOURCEDIR     = .
BUILDDIR      = _build
APIDIR        = source/api
PACKAGEDIR    = ../src/backend

# Internal imports:
#   - conf.py: Configuration for Sphinx documentation generation
#   - index.rst: Main entry point for documentation structure

# External imports:
#   - sphinx-build (sphinx): Command-line tool for building Sphinx documentation (sphinx-6.0+)
#   - sphinx-apidoc (sphinx): Tool for generating API documentation from Python modules (sphinx-6.0+)

.PHONY: help clean html dirhtml singlehtml latex latexpdf text man linkcheck doctest coverage apidoc

help: ## Display help information about available targets
	@echo "Please use 'make <target>' where <target> is one of"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

html: apidoc ## Build HTML documentation
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html $(SPHINXOPTS) $(SOURCEDIR)/$(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(SOURCEDIR)/$(BUILDDIR)/html."

dirhtml: apidoc ## Build HTML documentation with directory structure
	$(SPHINXBUILD) -b dirhtml $(SOURCEDIR) $(BUILDDIR)/dirhtml $(SPHINXOPTS) $(SOURCEDIR)/$(BUILDDIR)/dirhtml
	@echo
	@echo "Build finished. The HTML pages are in $(SOURCEDIR)/$(BUILDDIR)/dirhtml."

singlehtml: apidoc ## Build single HTML file documentation
	$(SPHINXBUILD) -b singlehtml $(SOURCEDIR) $(BUILDDIR)/singlehtml $(SPHINXOPTS) $(SOURCEDIR)/$(BUILDDIR)/singlehtml
	@echo
	@echo "Build finished. The HTML page is in $(SOURCEDIR)/$(BUILDDIR)/singlehtml."

latex: apidoc ## Build LaTeX documentation for PDF generation
	$(SPHINXBUILD) -b latex $(SOURCEDIR) $(BUILDDIR)/latex $(SPHINXOPTS) $(SOURCEDIR)/$(BUILDDIR)/latex
	@echo
	@echo "Build finished; the LaTeX files are in $(SOURCEDIR)/$(BUILDDIR)/latex."
	$(MAKE) -C $(BUILDDIR)/latex all

latexpdf: latex ## Build PDF documentation via LaTeX
	$(SPHINXBUILD) -b latex $(SOURCEDIR) $(BUILDDIR)/latex $(SPHINXOPTS) $(SOURCEDIR)/$(BUILDDIR)/latex
	@echo "Running LaTeX files through pdflatex..."
	$(MAKE) -C $(BUILDDIR)/latex all-pdf
	@echo "PDF build finished. The PDF files are in $(SOURCEDIR)/$(BUILDDIR)/latex."

text: apidoc ## Build plain text documentation
	$(SPHINXBUILD) -b text $(SOURCEDIR) $(BUILDDIR)/text $(SPHINXOPTS) $(SOURCEDIR)/$(BUILDDIR)/text
	@echo
	@echo "Build finished. The text files are in $(SOURCEDIR)/$(BUILDDIR)/text."

man: apidoc ## Build manual pages
	$(SPHINXBUILD) -b man $(SOURCEDIR) $(BUILDDIR)/man $(SPHINXOPTS) $(SOURCEDIR)/$(BUILDDIR)/man
	@echo
	@echo "Build finished. The manual pages are in $(SOURCEDIR)/$(BUILDDIR)/man."

clean: ## Clean build directory by removing all generated files
	rm -rf $(BUILDDIR)/*

linkcheck: ## Check all external links for integrity
	$(SPHINXBUILD) -b linkcheck $(SOURCEDIR) $(BUILDDIR)/linkcheck $(SPHINXOPTS) $(SOURCEDIR)/$(BUILDDIR)/linkcheck
	@echo "Link check complete; look for any errors in $(SOURCEDIR)/$(BUILDDIR)/linkcheck/output.txt."

doctest: ## Run doctests in the documentation
	$(SPHINXBUILD) -b doctest $(SOURCEDIR) $(BUILDDIR)/doctest $(SPHINXOPTS) $(SOURCEDIR)/$(BUILDDIR)/doctest
	@echo "Doctest complete; look for any errors in $(SOURCEDIR)/$(BUILDDIR)/doctest/output.txt."

coverage: ## Run coverage check on documentation
	$(SPHINXBUILD) -b coverage $(SOURCEDIR) $(BUILDDIR)/coverage $(SPHINXOPTS) $(SOURCEDIR)/$(BUILDDIR)/coverage
	@echo "Coverage report complete; look for details in $(SOURCEDIR)/$(BUILDDIR)/coverage/python.txt."

apidoc: ## Generate API documentation automatically from source code
	$(SPHINXAPIDOC) -o $(APIDIR) $(PACKAGEDIR)