# Local latexmk configuration for writings/00_Main/
# Overrides openout_any to allow writing .aux files in ../01_Chapters/
$pdflatex = 'pdflatex -cnf-line openout_any=a -synctex=1 -interaction=nonstopmode -file-line-error %O %S';
$pdf_mode = 1;
