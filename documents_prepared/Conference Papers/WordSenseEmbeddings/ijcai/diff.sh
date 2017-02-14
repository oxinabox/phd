latexdiff --exclude-safecmd="section,subsection,subsubsection" --exclude-textcmd="section,subsection,subsubsection" www_version.tex "AligningWordSenseEmbeddings.tex"  > diff.tex

sed -i 's|\\usepackage\[subpreambles=\w*\]{standalone}|%\\usepackage[subpreambles=true]{standalone}\n\\usepackage{environ}\n\\RenewEnviron{adjustbox}{Nothing here}|g' diff.tex
