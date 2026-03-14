#!/bin/bash

# Create output directory if it doesn't exist
mkdir -p out

# Compile LaTeX using latexmk
# -pdf: generate PDF
# -output-directory=out: put all auxiliary files and PDF in 'out/'
# -interaction=nonstopmode: don't stop on errors
# -bibtex: use bibtex for references
latexmk -pdf -output-directory=out -interaction=nonstopmode -bibtex Report1.tex

# Optional: Move the final PDF to the root for convenience
if [ -f "out/Report1.pdf" ]; then
    cp out/Report1.pdf .
    echo "Compilation successful. Report1.pdf is in the root directory."
else
    echo "Compilation failed. Check the logs in out/Report1.log."
fi
