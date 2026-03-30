# Makefile for rendering FunctorFlow.jl vignettes
#
# Usage:
#   make all        — render all vignettes to md, html, pdf
#   make html       — render all vignettes to html only
#   make pdf        — render all vignettes to pdf only
#   make md         — render all vignettes to gfm markdown only
#   make 01         — render vignette 01 only (all formats)
#   make clean      — remove rendered outputs
#
# Prerequisites:
#   - quarto (https://quarto.org)
#   - Julia with FunctorFlow.jl project
#   - STIX Two Text font (for PDF)
#   - JuliaMono font (for code in PDF, optional)

SHELL := /bin/bash

# Discover all vignette directories
VIGNETTE_DIRS := $(sort $(wildcard [0-9][0-9]-*))

# Find the .qmd file in each directory
qmd_file = $(wildcard $(1)/*.qmd)

# Quarto flags
QUARTO := quarto
QUARTO_FLAGS := --execute
JULIA_PROJECT_DIR := $(realpath ..)

# Export JULIA_PROJECT for Quarto's Julia engine (QuartoNotebookRunner)
export JULIA_PROJECT := $(JULIA_PROJECT_DIR)

.PHONY: all html pdf md clean $(VIGNETTE_DIRS)

all: $(VIGNETTE_DIRS)

html: FORMATS := html
html: $(VIGNETTE_DIRS)

pdf: FORMATS := pdf
pdf: $(VIGNETTE_DIRS)

md: FORMATS := gfm
md: $(VIGNETTE_DIRS)

# Per-directory targets
$(VIGNETTE_DIRS):
	@echo "=== Rendering $@ ==="
	@qmd=$$(ls $@/*.qmd 2>/dev/null | head -1); \
	if [ -z "$$qmd" ]; then \
		echo "  No .qmd file found in $@, skipping"; \
	elif [ -n "$(FORMATS)" ]; then \
		cd $@ && $(QUARTO) render $$(basename $$qmd) --to $(FORMATS) $(QUARTO_FLAGS); \
	else \
		cd $@ && $(QUARTO) render $$(basename $$qmd) $(QUARTO_FLAGS); \
	fi

# Shortcut targets: make 01, make 02, etc.
01: 01-getting-started
02: 02-dsl-macros
03: 03-kan-extensions
04: 04-block-library
05: 05-composition
06: 06-obstruction-loss
07: 07-lux-neural-backend
08: 08-universal-constructions
09: 09-causal-semantics
10: 10-topos-foundations
11: 11-canonical-v1-example
12: 12-sudoku-constraints
13: 13-predict-detach
14: 14-jepa-world-models
15: 15-ket-language-model
16: 16-kan-duality
17: 17-causal-jepa
18: 18-neurosymbolic-pipelines

clean:
	@for dir in $(VIGNETTE_DIRS); do \
		echo "Cleaning $$dir"; \
		rm -f $$dir/*.html $$dir/*.pdf $$dir/*.md; \
		rm -rf $$dir/*_files; \
	done
	@echo "Done."
