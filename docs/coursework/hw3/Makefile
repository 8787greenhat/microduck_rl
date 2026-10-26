# HW3 report: `make` builds build/progress.pdf with XeLaTeX (two passes for
# cross-references). `make deps` prints the Ubuntu/WSL packages it needs.

TEX     := progress.tex
BUILD   := build
PDF     := $(BUILD)/progress.pdf
FIGURES := $(wildcard figures/*)
XELATEX := xelatex -interaction=nonstopmode -halt-on-error -output-directory=$(BUILD)

.PHONY: all view clean deps

all: $(PDF)

$(PDF): $(TEX) $(FIGURES) | $(BUILD)
	@command -v xelatex >/dev/null || { echo "xelatex not found. Install it with:"; $(MAKE) --no-print-directory deps; exit 1; }
	@for pass in 1 2; do \
	  echo "xelatex pass $$pass"; \
	  $(XELATEX) $(TEX) > $(BUILD)/pass$$pass.out 2>&1 || { \
	    echo "xelatex failed; end of $(BUILD)/progress.log:"; tail -n 25 $(BUILD)/progress.log; exit 1; }; \
	done
	@grep -E "Overfull|undefined|Missing character" $(BUILD)/progress.log || true
	@grep -o "Output written on .*" $(BUILD)/pass2.out

$(BUILD):
	@mkdir -p $@

view: $(PDF)
	@xdg-open $(PDF) >/dev/null 2>&1 || wslview $(PDF) >/dev/null 2>&1 || echo "Open $(CURDIR)/$(PDF)"

clean:
	rm -rf $(BUILD)

deps:
	@echo "sudo apt install texlive-xetex texlive-lang-cjk texlive-pictures texlive-latex-extra fonts-texgyre fonts-arphic-bsmi00lp"
