\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{komatufte}[2017/07/05 A Hacked Up Tufte-like Koma book]
% Based on https://tex.stackexchange.com/questions/333111/how-to-align-the-baseline-of-margin-notes-in-latex-with-the-main-text

\PassOptionsToPackage{hyphens}{url}

% General setup
\RequirePackage{leading}
\RequirePackage{calc}
\RequirePackage{xparse}


\RequirePackage{xcolor}
\definecolor{blue}{rgb}{0.2, 0.2, 1.0}


\LoadClass[
a4paper,
twoside,
titlepage,
fontsize = 14pt,
parskip = full,
%headings = small,
index = totoc,
listof = totoc,
bibliography = totoc,
numbers = noenddot,
appendixprefix = true,
captions = nooneline
]{scrbook}

%%%%%%%%%%%%%%%
% Page Layout





\newcommand\pagedivisions{14}
\newlength{\completemargin}
\setlength{\completemargin}{6cm}

\RequirePackage{geometry}
\geometry{
	marginpar = 5.5cm,
	inner  = 21cm /  \pagedivisions,
	outer  = 21cm / \pagedivisions,
	top    = 29.7cm / \pagedivisions,
	bottom = 29.7cm / \pagedivisions,
	heightrounded,
}
\savegeometry{basegeo} % Reset to this geometry for title pages etc

\geometry{
	marginpar = 5.5cm,
	inner  = 21cm /  \pagedivisions,
	outer  = 21cm / \pagedivisions + \completemargin,
	top    = 29.7cm / \pagedivisions,
	bottom = 29.7cm / \pagedivisions,
	heightrounded,
}
\savegeometry{contentgeo} % use this geometry for normal pages



% Use scrlayer-scrpage, so that package notecolumn works
\RequirePackage[
%	headwidth = 21cm - 21cm / \pagedivisions * 3 : 1cm,
%	footwidth = 21cm - 21cm / \pagedivisions * 3 : 1cm,
%	plainheadsepline,
%	headsepline = 1pt,
%	plainfootsepline,
]{scrlayer-scrpage}


\newlength{\notescolwidth}
\setlength{\notescolwidth}{\completemargin - \marginparsep}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Basis of asides

% Font Style for note column
\let\ocolor\color % Protect it, see: https://tex.stackexchange.com/a/415272/5834
\NewDocumentCommand{\asidefont}{}{\small\sffamily\color{blue}}


% Set up the note column
\RequirePackage{etoolbox}
\RequirePackage{xpatch}
\AtEndPreamble{
	\RequirePackage{scrlayer-notecolumn} %must be loaded ``Lastest''
	\RedeclareNoteColumn[
		normalmarginpar,
		width = \notescolwidth,
		font = \asidefont,
	]{marginpar} %default
}





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%

\RequirePackage{ragged2e}
%\newcommand{\asideheading}[1]{{\bfseries\raggedright\parfillskip=-\rightskip #1\par}}
\newcommand{\asideheading}[1]{{\bfseries\setlength{\RaggedRightParfillskip}{-\RaggedRightRightskip}\RaggedRight #1\par}}

	
\NewDocumentCommand{\aside}{o +m}{%
	\IfNoValueTF{#1}%
	{\makenote*{#2\bigskip}}%
	{\makenote*{\asideheading{#1} #2\bigskip}}%
}
\NewDocumentCommand{\asideunsafe}{o +m}{%
	\IfNoValueTF{#1}%
	{\makenote{#2\bigskip}}%
	{\makenote{\asideheading{#1} #2\bigskip}}%
}

\NewDocumentCommand{\asidefig}{o +m}{%
	\makenote*{%
		\centering%
		#2%
		\clearcaptionsetup{figure}%
		\let\normalcolor\relax
		\captionof{figure}{ #1}%
		\bigskip%
	}%

}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Captions
% https://tex.stackexchange.com/questions/352052/how-to-insert-vspaces-into-scrlayer-notecolumn-marginnote

\RequirePackage[]{caption}%
\captionsetup{compatibility=false}%

\DeclareCaptionFont{asidefont}{\asidefont}
\DeclareCaptionFormat{sidenote}{\asideunsafe{\vskip 0.5\baselineskip #1#2#3}}%
\DeclareCaptionStyle{goodstyle}{parskip=0pt,skip=0pt,singlelinecheck=off, labelfont+={bf}, font+={asidefont}}%
\captionsetup{style=goodstyle}

\captionsetup[figure]{format=sidenote}%
\captionsetup[table]{format=sidenote}%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Bibliography Commands
%
\RequirePackage[
	backend=biber,
	url=false,  
	style=alphabetic,
	citestyle=authoryear,
	bibstyle=authoryear,
	maxbibnames=99,
	dashed=false,
	refsection=chapter
]{biblatex}


\newcommand{\sidecite}[1]{%
	{\aside{\textcite{#1}, \citetitle{#1}}}%
}

\newcommand{\pcite}[1]{%
	\parencite{#1}%
	\sidecite{#1}%
}
\newcommand{\tcite}[1]{%
	\textcite{#1}%
	\sidecite{#1}%
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
\renewcommand*\dictumwidth{0.8\textwidth}
\renewcommand*\dictumauthorformat[1]{--- #1}
\renewcommand*\dictumrule{}


%%%%%%%%%%%%%%%%%%%%%%
% A Chapter abstract
\RequirePackage{environ}

\NewEnviron{abstract}{%
	\addtocontents{toc}{\small\BODY\par}%
	\rightskip1in{\BODY}%
}