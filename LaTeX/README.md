# Compilation du rapport

Depuis la racine du dépôt, avec une distribution TeX comprenant `latexmk` et `biber` :

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error LaTeX/plan_revise_15_pages.tex
```

La configuration `.latexmkrc` utilise le dossier du document comme répertoire de
travail. `latexmk` lance automatiquement PDFLaTeX, Biber puis les passes LaTeX
nécessaires pour résoudre les citations et la table des matières.

La bibliographie est stockée dans `ref.bib`, chargée par
`\addbibresource{ref.bib}` et affichée par `\printbibliography`. Ajouter une
référence au texte avec `\cite{cle}` : seules les références citées sont imprimées.
Dans un éditeur, utiliser une recette `latexmk` avec Biber, ou la séquence
PDFLaTeX → Biber → PDFLaTeX → PDFLaTeX depuis le dossier `LaTeX`.

Les fichiers `.tex`, `.bib` et `.pdf` sont conservés dans Git ; les fichiers
auxiliaires de compilation sont ignorés.
