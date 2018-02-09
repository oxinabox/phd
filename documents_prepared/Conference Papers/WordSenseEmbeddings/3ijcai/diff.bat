 latexdiff -V --math-markup=0 --exclude-safecmd="section,subsection,subsubsection" --exclude-textcmd="section,subsection,subsubsection" draft4.tex "AligningWordSenseEmbeddings.tex"  > diff.tex
 
 GOTO LABEL
 
 %\usepackage[subpreambles=true]{standalone}
 \usepackage{environ}
 \RenewEnviron{adjustbox}{[[]]FIGURE OR TABLE EXCLUDED FROM DIFF]]}
 :LABEL