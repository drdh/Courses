OUT = main

all: $(OUT) clean

$(OUT):  $(wildcard *.tex *.bib *.cls *.png)
	xelatex $@
	xelatex $@

clean:
	rm $(wildcard *~ *.aux *.dvi *.bbl *.blg *.log *.out *.toc *.lof *.lot)

clean-pdf:
	rm main.pdf