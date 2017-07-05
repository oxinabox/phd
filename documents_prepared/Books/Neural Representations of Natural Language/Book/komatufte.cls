\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{komatufte}[2017/07/05 A Hacked Up Tufte-like Koma book]
% Based on https://tex.stackexchange.com/questions/333111/how-to-align-the-baseline-of-margin-notes-in-latex-with-the-main-text

% General setup
\RequirePackage{blindtext}
\RequirePackage{leading}
\RequirePackage{calc}
\RequirePackage{color}
\definecolor{blue}{rgb}{0.2, 0.2, 1.0}
\newcommand\pagedivisions{14}

% Page Layout
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
\geometry{
  inner  = 21cm /  \pagedivisions,
  outer  = 21cm / \pagedivisions + \completemargin,
  top    = 29.7cm / \pagedivisions,
  bottom = 29.7cm / \pagedivisions,
  heightrounded
}

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
  font = \raggedleft\small\sffamily\color{blue}
]{notes}


\newcommand{\aside}[1]{
	\ifvmode\leavevmode\fi
	\makenote[notes]{#1}
}