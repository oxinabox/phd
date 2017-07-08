\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{komatufte}[2017/07/05 A Hacked Up Tufte-like Koma book]
% Based on https://tex.stackexchange.com/questions/333111/how-to-align-the-baseline-of-margin-notes-in-latex-with-the-main-text

% General setup
\RequirePackage{blindtext}
\RequirePackage{leading}
\RequirePackage{calc}

\RequirePackage{color}
\definecolor{blue}{rgb}{0.2, 0.2, 1.0}

%\RequirePackage{scrextend}


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
\savegeometry{basegeo}
\geometry{
  marginpar = 5.5cm,
  inner  = 21cm /  \pagedivisions,
  outer  = 21cm / \pagedivisions + \completemargin,
  top    = 29.7cm / \pagedivisions,
  bottom = 29.7cm / \pagedivisions,
  heightrounded
}
\savegeometry{contentgeo}


%%%%%
% Make chapter titles not restricted by main width
\renewcommand*{\chapterlinesformat}[3]{%
	\makebox[\textwidth][l]{% avoid overfull \hbox
		\parbox[t]{\dimexpr\textwidth+\marginparsep+\marginparwidth}{% use more width
			\raggedchapter
			\@hangfrom{#2}{#3}%
		}%
	}%
}

%%%%%

% Use scrlayer-scrpage, so that package notecolumn works
\RequirePackage[
  headwidth = 21cm - 21cm / \pagedivisions * 3 : 0cm,
  footwidth = 21cm - 21cm / \pagedivisions * 3 : 0cm,
  plainheadsepline,
  headsepline = 1pt,
  plainfootsepline,
]{scrlayer-scrpage}

% Set up the note column
\RequirePackage{scrlayer-notecolumn}
\newlength{\notescolwidth}
\setlength{\notescolwidth}{\completemargin - \marginparsep}

\DeclareNewNoteColumn[
  normalmarginpar,
  width = \notescolwidth,
  font = \ifthispageodd{\raggedleft}{\raggedright}\small\sffamily\color{blue}
]{notes}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
\newcommand{\aside}[1]{
	\ifvmode\leavevmode\fi
	\makenote[notes]{#1}
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Bibliography Commands

\newcommand{\tcite}[1]{
	\textcite{#1}
	\aside{\fullcite{#1}}
} 


%%%%%%%%%%%%%%%%%%%%%%%%
% customize dictum format:
\setkomafont{dictumtext}{\itshape\small\sffamily}
\setkomafont{dictumauthor}{\normalfont\small\sffamily}
\renewcommand*\dictumwidth{0.8\linewidth}
\renewcommand*\dictumauthorformat[1]{--- #1}
\renewcommand*\dictumrule{}
