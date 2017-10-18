\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{komatufte}[2017/07/05 A Hacked Up Tufte-like Koma book]
% Based on https://tex.stackexchange.com/questions/333111/how-to-align-the-baseline-of-margin-notes-in-latex-with-the-main-text



% General setup
\RequirePackage{etoolbox}

\RequirePackage{leading}
\RequirePackage{calc}
\RequirePackage{xparse}


\RequirePackage{color}
\definecolor{blue}{rgb}{0.2, 0.2, 1.0}

%%%%%%%%%%%%%%%
% Page Layout

\newcommand\pagedivisions{14}
\LoadClass[
  a4paper,
  twoside,
  titlepage,
  fontsize = 14pt,
  parskip = full,
  headings = small,
  index = totoc,
  listof = totoc,
  bibliography = totoc,
  numbers = noenddot,
  appendixprefix = true,
  captions = nooneline
]{scrbook}

\newlength{\completemargin}
\setlength{\completemargin}{6cm}

\RequirePackage{geometry}
\savegeometry{basegeo} % Reset to this geometry for title pages etc
\geometry{
  marginpar = 5.5cm,
  inner  = 21cm /  \pagedivisions,
  outer  = 21cm / \pagedivisions + \completemargin,
  top    = 29.7cm / \pagedivisions,
  bottom = 29.7cm / \pagedivisions,
  heightrounded
}
\savegeometry{contentgeo} % use this geometry for normal pages



% Use scrlayer-scrpage, so that package notecolumn works
\RequirePackage[
  headwidth = 21cm - 21cm / \pagedivisions * 3 : 0cm,
  footwidth = 21cm - 21cm / \pagedivisions * 3 : 0cm,
  plainheadsepline,
  headsepline = 1pt,
  plainfootsepline,
]{scrlayer-scrpage}

\newlength{\notescolwidth}
\setlength{\notescolwidth}{\completemargin - \marginparsep}

% Set up the note column
\AtEndPreamble{\RequirePackage{scrlayer-notecolumn} %must be loaded ``Lastest''
\RedeclareNoteColumn[
normalmarginpar,
width = \notescolwidth,
font = \ifthispageodd{\raggedleft}{\raggedright}\small\sffamily\color{blue}
]{marginpar} %default
}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Make chapter titles not restricted by main width
\renewcommand*{\chapterlinesformat}[3]{%
	\makebox[\textwidth][l]{% avoid overfull \hbox
		\parbox[t]{\dimexpr\textwidth+\marginparsep+\marginparwidth}{% use more width
			\raggedchapter
			\@hangfrom{#2}{#3}%
		}%
	}%
}


%%%%%%%%%%%%%%%%%%%%%%%%
% customize dictum format:
\setkomafont{dictumtext}{\itshape\small\sffamily}
\setkomafont{dictumauthor}{\normalfont\small\sffamily}
\renewcommand*\dictumwidth{0.8\linewidth}
\renewcommand*\dictumauthorformat[1]{--- #1}
\renewcommand*\dictumrule{}


%%%%%%%%%%%%%%%%%%%%%%
% A Chapter abstract
\RequirePackage{environ}

\NewEnviron{abstract}{%
	\addtocontents{toc}{\small\BODY\par}%
	\rightskip1in\textsc{\BODY}%
}



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%


\NewDocumentCommand{\aside}{o +m}{
	\IfNoValueTF{#1}
	{\makenote*{#2\bigskip}}
	{\makenote*{{\textbf{#1}\\}#2\bigskip}}
}
\NewDocumentCommand{\asideunsafe}{o +m}{
	\IfNoValueTF{#1}
	{\makenote{#2\bigskip}}
	{\makenote{{\textbf{#1}\\}#2\bigskip}}
}

\NewDocumentCommand{\asidefig}{o +m}{
	\makenote*{%
		#2
		\clearcaptionsetup{figure}
		\captionof{figure}{#1}
		\bigskip
	}
}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Captions
% https://tex.stackexchange.com/questions/352052/how-to-insert-vspaces-into-scrlayer-notecolumn-marginnote

\RequirePackage[]{caption}%
\captionsetup{compatibility=false}%

\DeclareCaptionFormat{sidenote}{\asideunsafe{\vskip 0.5\baselineskip #1#2#3}}%
\DeclareCaptionStyle{goodstyle}{parskip=0pt,skip=0pt,singlelinecheck=off, labelfont=bf}%
\captionsetup{style=goodstyle}

\captionsetup[figure]{format=sidenote}%
\captionsetup[table]{format=sidenote}%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Bibliography Commands
%
\RequirePackage[
	backend=bibtex,
	url=false,  
	style=alphabetic,
	citestyle=authoryear,
]{biblatex}


\newcommand{\sidecite}[1]{
	%\ifentryseen{#1}
	{\aside{\textcite{#1}, \citetitle{#1}}}%
	%{}%do nothing if already shown this chapter	
}

\newcommand{\pcite}[1]{
	\parencite{#1}%
	\sidecite{#1}%
}
\newcommand{\tcite}[1]{
	\textcite{#1}%
	\sidecite{#1}%
}

