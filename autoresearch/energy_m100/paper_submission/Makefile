# Build the NeurIPS 2026 paper.
# Requires pdflatex + the neurips_2026.sty official template (or use the
# bundled geometry fallback which compiles with plain pdflatex).
#
# Usage:  make        -> neurips_2026.pdf
#         make clean  -> remove aux files

PDF = neurips_2026.pdf
TEX = neurips_2026.tex

all: $(PDF)

$(PDF): $(TEX) figures/*.png
	pdflatex $(TEX)
	pdflatex $(TEX)

clean:
	rm -f *.aux *.log *.out *.bbl *.blg $(PDF)
