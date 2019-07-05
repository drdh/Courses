OUT = main

all:  $(OUT) clean

$(OUT):  $(wildcard *.tex *.bib *.cls *.eps *.jpg)
	xelatex $@
	bibtex $@
	xelatex $@
	xelatex $@
   
clean:
	rm -f *~ *.aux *.dvi *.bbl *.blg *.log *.out *.toc 
##	evince main.pdf &
clean-pdf:
	rm *.pdf
