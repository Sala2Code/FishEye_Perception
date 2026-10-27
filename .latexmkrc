# Compiler en PDF et travailler dans le dossier du fichier .tex.
# latexmk lance automatiquement Biber lorsque biblatex le demande.
$pdf_mode = 1;
$do_cd = 1;
# Autoriser la conversion des SVG par Inkscape (package svg).
$pdflatex = 'pdflatex -shell-escape %O %S';
