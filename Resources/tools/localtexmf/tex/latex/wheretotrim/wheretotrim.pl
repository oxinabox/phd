#! /usr/bin/env perl

########################################
# Identify opportunities for reducing  #
# the page count of a LaTeX document   #
#                                      #
# By Scott Pakin <scott+wtt@pakin.org> #
########################################

use File::Basename;
use File::Temp qw(tempfile);
use Getopt::Long;
use POSIX;
use Pod::Usage;
use warnings;
use strict;

# Define some global variables.
my $progname = basename $0;   # Name of this program
my $logfile;                  # LaTeX-generated log file
my $verbosity = 1;            # Level of output verbosity
my $allpages = 0;             # 1=report changes needed for all pages; 0=any page
my @latexcmd;                 # Complete command to run LaTeX
my $ltxfile;                  # Name of input file
my $colsperpage = 1;          # Number of columns per page (1 or 2)
my %column2page;              # Map from absolute column number to {page, column}
my $debugexp;                 # Typeset using an expansion of <page>,<column>,<expansion lines> for debugging
our $VERSION = "1.0";         # Version number of this program

# Define a subroutine that replaces a file name with its base name and
# (optionally) new suffix.
sub basename_newsuffix ($;$)
{
    my ($fname, $newsuffix) = @_;
    my ($basename, undef, undef) = fileparse($fname, qr/\.[^.]*/);
    $newsuffix = "" if !defined $newsuffix;
    return $basename . $newsuffix;
}

# Define a subroutine to create a temporary LaTeX file that modifies a
# few LaTeX commands then loads the user's document.  The subroutine
# returns the name of the temporary file.
sub create_latex_file ($$$)
{
    my ($columntoexpand, $columnexpandlines, $extrafullcolumns) = @_;
    my ($modltx, $modltxfile) = tempfile("wtt-XXXXXX",
                                         TMPDIR => 1,
                                         SUFFIX => ".tex",
                                         UNLINK => 1);
    print $modltx "\\RequirePackage[column=$columntoexpand,expansion=$columnexpandlines,extracols=$extrafullcolumns]{wheretotrim}\n";
    print $modltx "\\PassOptionsToPackage{draft}{hyperref}\n";   # Avoid "\pdfendlink ended up in different nesting level than \pdfstartlink" errors.
    print $modltx "\\input{$ARGV[$#ARGV]}\n";
    close $modltx;
    return $modltxfile;
}

# Define a subroutine to run LaTeX on a given filename.
sub run_latex ($$$$)
{
    # Add some additional arguments to the LaTeX command.
    my ($modltxfile, $columntoexpand, $columnexpandlines, $extrafullcolumns) = @_;
    my $jobname = basename_newsuffix($ltxfile);
    @latexcmd = (@ARGV[0..$#ARGV-1], "-jobname=$jobname", $modltxfile);

    # Run LaTeX.
    if ($verbosity == 1) {
        if ($columntoexpand == 0) {
            print "Compiling $ltxfile normally";
            if ($extrafullcolumns > 0) {
                printf ", but with %s column%s of padding", $extrafullcolumns, $extrafullcolumns == 1 ? "" : "s";
            }
            print " ... ";
        }
        elsif ($colsperpage == 1) {
            my ($page, $col) = @{$column2page{$columntoexpand}};
            printf "Compiling %s with page %d expanded by %d line%s ... ",
            $ltxfile, $page, $columnexpandlines, $columnexpandlines == 1 ? "" : "s";
        }
        else {
            my ($page, $col) = @{$column2page{$columntoexpand}};
            printf "Compiling %s with page %d, column %d expanded by %d line%s ... ",
            $ltxfile, $page, $col, $columnexpandlines, $columnexpandlines == 1 ? "" : "s";
        }
    }
    elsif ($verbosity > 1) {
        print "Running @latexcmd\n";
    }
    open(LATEX, "-|", "sh", "-c", 'echo X | "$@" 2>&1', "--", $latexcmd[0], @latexcmd[1..$#latexcmd]) || die;   # Redirect stderr to stdout as we run.
    while (my $oneline = <LATEX>) {
        print $oneline if $verbosity > 1;
    }
    close LATEX;
    my $errcode = $?;
    if ($verbosity == 1) {
        print $errcode == 0 ? "done.\n" : "failed.\n";
    }
    elsif ($verbosity > 1) {
        print "Finished running.\n";
    }
    return $errcode;
}

# Define a subroutine to process a log file and return various data
# extracted from it.
sub process_log_file ($$$)
{
    my ($columntoexpand, $columnexpandlines, $extrafullcolumns) = @_;
    my %column_map;

    # Extract wheretotrim information lines and the final page count.
    print "Processing $logfile ... " if $verbosity > 0;
    my ($numpages, $baselineskip, $textheight) = (0, 0, 0);
    open(LOGFILE, "<", $logfile) || die "${progname}: Failed to open $logfile ($!)\n";
    my $infostr = "Package wheretotrim Info";
    while (my $oneline = <LOGFILE>) {
        $baselineskip = $1+0 if $oneline =~ /^$infostr: Baseline skip: ([\d.]+)pt/;
        $textheight = $1+0 if $oneline =~ /^$infostr: Text height: ([\d.]+)pt/;
        $column_map{$1} = [$2, $3] if $oneline =~ /^$infostr: Column (\d+) is on page (\d+) \((.*)\) on input line/;
        $numpages = $1 if $oneline =~ /^Output written on.*\((\d+) page/;
    }
    close LOGFILE;
    $numpages-- if $extrafullcolumns > 0;
    if ($verbosity > 0) {
        printf "done (%d page%s).\n",
        $numpages, $numpages == 1 ? "" : "s",
    }
    return ($numpages, $baselineskip, $textheight, \%column_map);
}

# Define a subroutine to run LaTeX and return a page count and other
# information.
sub latex_page_count ($$$)
{
    # LaTeX wrapper scripts might not like being given LaTeX code on
    # the command line.  We therefore create a temporary file that
    # prepares LaTeX for programmatically modifying column heights.
    my ($columntoexpand, $columnexpandlines, $extrafullcolumns) = @_;
    my $modltxfile = create_latex_file($columntoexpand, $columnexpandlines, $extrafullcolumns);

    # Run latex on the temporary file.
    my $errcode = run_latex($modltxfile, $columntoexpand, $columnexpandlines, $extrafullcolumns);
    unlink $modltxfile;

    # Process the log file.
    return (0, undef, undef, undef) if $errcode != 0;
    return process_log_file($columntoexpand, $columnexpandlines, $extrafullcolumns);
}

###########################################################################

# Parse the command line.
my $wanthelp = 0;
my $wantversion = 0;
Getopt::Long::Configure("require_order");
GetOptions("h|help"       => \$wanthelp,
           "V|version"    => \$wantversion,
           "a|allpages"   => \$allpages,
           "l|logfile=s"  => \$logfile,
           "v|verbose+"   => \$verbosity,
           "d|debug=s"    => \$debugexp,
           "q|quiet"      => sub {$verbosity = 0})
    || pod2usage(-exitval => 2);
if ($wantversion) {
    print "wheretotrim $VERSION\n";
    exit 0;
}
pod2usage(-verbose => $verbosity,
          -exitval => 1) if $wanthelp;
pod2usage(-message => "${progname}: A latex command must be specified",
          -exitval => 2) if $#ARGV == -1;
$ltxfile = basename($ARGV[$#ARGV]);
$logfile = basename_newsuffix($ARGV[$#ARGV], ".log") if !defined $logfile;

# Determine the document's baseline characteristics.
my ($basepages, $baselineskip, $textheight, $c2p_ptr) = latex_page_count 0, 0, 0;
die "${progname}: Failed to build $ltxfile\n" if $basepages == 0;
%column2page = %$c2p_ptr;
print "\n" if $verbosity > 0;

# Map an absolute column to a page and column number.
my $prevpage = 0;
foreach my $col (sort {$a <=> $b} keys %column2page) {
    my ($pagenum, $pagename) = @{$column2page{$col}};
    if ($pagenum == $prevpage) {
        $column2page{$col} = [$pagenum, 2, $pagename];
        $colsperpage = 2;
    }
    else {
        $column2page{$col} = [$pagenum, 1, $pagename];
    }
    $prevpage = $pagenum;
}

# If we were given a page, column, and expansion, typeset the document
# with those parameters and exit.
if (defined $debugexp) {
    die "${progname}: Failed to parse \"$debugexp\" into {page, column, expansion}\n" if $debugexp !~ /^(\d+)\D+(\d+)\D+(\d+)$/;

    # Convert page and column number to absolute column number.
    my ($target_page, $target_col, $expansion) = ($1, $2, $3);
    my $testcol;
    while (my ($abscol, $page_col) = each %column2page) {
        if ($target_page == $page_col->[0] && $target_col == $page_col->[1]) {
            $testcol = $abscol;
            last;
        }
    }
    die "${progname}: Failed to map page $target_page, column $target_col to an absolute column number\n" if !defined $testcol;

    # Enlarge the given page.
    my ($numpages, undef) = latex_page_count $testcol, $expansion, $colsperpage;
    print "\n" if $verbosity > 0;
    latex_page_count $testcol, $expansion, 0;   # Run again without appending any extra columns.
    print "\n" if $verbosity > 0;
    print "Expanding page $target_page, column $target_col by $expansion lines ";
    if ($numpages == $basepages) {
        print "does not reduce the page count below $numpages pages.\n";
    }
    else {
        print "reduces the page count from $basepages pages to $numpages pages.\n";
    }
    exit 0;
}

# Determine columns for which no amount of expansion will reduce the
# page count.
my $maxexpansion = int($textheight/$baselineskip + 1);
my @complete = (0, 0+keys %column2page);   # Fraction complete (numerator and denominator)
foreach my $expcol (sort {$a <=> $b} keys %column2page) {
    my ($numpages, undef) = latex_page_count $expcol, $maxexpansion, $colsperpage;
    if ($verbosity > 0) {
        $complete[0]++;
        printf "Trial runs are %.1f%% complete.\n\n", 100.0*$complete[0]/$complete[1];
    }
    delete $column2page{$expcol} if $numpages > 0 && $numpages == $basepages;
}

# Keep expanding a page by greater and greater amounts until we reduce
# our page count.
my %col2savings;     # Map from an absolute column to an {expansion, page count} tuple.
my $target_num_cols = $allpages ? (keys %column2page) : 1;   # Minimum number of columns for which to find an expansion amount
my $minexpansion;    # Minimum value of the above that saves a page
@complete = (0, $maxexpansion*keys %column2page);
foreach my $expansion (1 .. $maxexpansion) {
    # Expand each column in turn.
    foreach my $expcol (sort {$a <=> $b} keys %column2page) {
        $complete[0]++;
        next if defined $col2savings{$expcol};   # Already finished
        next if $column2page{$expcol}->[0] == $basepages && $column2page{$expcol}->[1] == 2;   # Second column on the last page
        my ($numpages, undef) = latex_page_count $expcol, $expansion, $colsperpage;
        if ($numpages > 0 && $numpages < $basepages) {
            $col2savings{$expcol} = [$expansion, $numpages];
            $minexpansion = $expansion if !defined $minexpansion;
        }
        if ($verbosity > 0) {
            printf "Execution is %.1f%% complete.\n\n", 100.0*$complete[0]/$complete[1];
        }
    }
    last if keys %col2savings >= $target_num_cols;    # Success
}

# Restore the document to its original form.
run_latex $ltxfile, 0, 0, 0;
printf "Execution is 100.0%% complete.\n\n" if $verbosity > 0;

# Output the space savings.
if (keys %col2savings == 0) {
    printf "It does not appear possible to reduce the page count from %d to %d\n",
    $basepages, $basepages-1;
    print "by removing any amount of text from any single column.\n\n";
    exit 0;
}
printf "To reduce the page count from %d to %d, do %s following:\n\n",
    $basepages, $basepages-1, keys %col2savings == 1 ? "the" : "any of the";
foreach my $abscol (sort {$a <=> $b} keys %col2savings) {
    my ($expansion, $numpages) = @{$col2savings{$abscol}};
    my ($page, $col, $pagename) = @{$column2page{$abscol}};
    print "  * Reduce page $page";
    print " (\"$pagename\")" if $pagename ne $page;
    print ", column $col" if $colsperpage > 1;
    printf " by %d %s", $expansion, $expansion == 1 ? "line" : "lines";
    if ($numpages < $basepages - 1) {
        printf " (produces %d %s)", $numpages, $numpages == 1 ? "page" : "pages";
    }
    print ".\n";
}
print "\n";
my $minpoints = $minexpansion*$baselineskip;
printf "Note: %d lines = %.1f\" = %.1f cm = %.1f%% of the %s height\n",
    $minexpansion, $minpoints/72.27, $minpoints/28.45,
    100.0*$minpoints/$textheight, $colsperpage == 1 ? "page" : "column";

###########################################################################

__END__

=head1 NAME

wheretotrim - Help reduce the page count of a LaTeX document

=head1 SYNOPSIS

wheretotrim
[B<--verbose> | B<--quiet>]
[B<--allpages>]
[B<--debug>=I<page>,I<column>,I<lines>]
I<latex command>

wheretotrim [B<--verbose>] B<--help>|B<--version>

=head1 DESCRIPTION

B<wheretotrim> is a tool to help LaTeX users reduce their document's
page count.  It is intended to be used with documents that exceed a
publisher's specified page-length limitation by a small amount (much
less than a full column or page).  B<wheretotrim> operates by building
the document repeatedly, successively expanding each column on each
page by one line height to mimic reducing the amount of text in that
column by an equivalent amount.  If doing so does not reduce the page
count, B<wheretotrim> repeats the process with two line heights'
expansion of each column, then three, and so forth until it expands
each column in turn by the full height of the column.  The following
is some sample output:

    To reduce the page count from 10 to 9, do any of the following:

      * Reduce page 9, column 1 by 12 lines.
      * Reduce page 9, column 2 by 12 lines.
      * Reduce page 10, column 1 by 12 lines.

    Note: 12 lines = 2.4" = 6.1 cm = 26.8% of the column height

That is, reducing either column on S<page 9> or the first column on
S<page 10> by 12 lines is the most expedient way to reduce the
document's page count.  More than S<12 lines> would need to be cut on
other columns and other pages to achieve the same effect.

=head1 OPTIONS

B<wheretotrim> accepts the following command-line options:

=over 4

=item B<-a>, B<--allpages>

Perform enough extra runs of B<latex> to report the amount of space
that must be trimmed from I<each> column or page to reduce page count,
not just the columns or pages to which the page count is the most
sensitive.

=item B<-v>, B<--verbose>

Display the output of each run of B<latex>.  This is useful for
troubleshooting and to help monitor the progress of long B<latex>
runs.

=item B<-q>, B<--quiet>

Suppress progress updates and output only the final report.

=item B<-d> I<page>,I<column>,I<lines>, B<--debug>=I<page>,I<column>,I<lines>

Debug B<wheretotrim>'s execution by expanding page I<page>, column
I<column> by I<lines> line heights and leaving the B<latex> output in
that state.

=item B<-h>, B<--help>

Summarize usage information and exit.  These may be used with
B<--verbose> to display more extended documentation.

=item B<-V>, B<--version>

Display B<wheretotrim>'s version number and exit.

=back

In addition to the preceding options, B<wheretotrim> requires a
I<latex command> argument that specifies how to build the document.

=head1 EXAMPLES

For the most basic usage, simply provide a B<latex> command to run:

    wheretotrim latex myfile.tex

or, for example,

    wheretotrim pdflatex myfile.tex

B<wheretotrim> executes the specified command a large number of times
and finally terminates with a report resembling the following:

    To reduce the page count from 10 to 9, do any of the following:

      * Reduce page 9, column 1 by 12 lines.
      * Reduce page 9, column 2 by 12 lines.
      * Reduce page 10, column 1 by 12 lines.

    Note: 12 lines = 2.4" = 6.1 cm = 26.8% of the column height

To ask B<wheretotrim> to report how much space needs to be trimmed on
each column and page to reduce the total page count, specify the
B<--allpages> option:

    wheretotrim --allpages latex myfile.tex

The output now looks like the following:

    To reduce the page count from 10 to 9, do any of the following:

      * Reduce page 1, column 1 by 13 lines.
      * Reduce page 1, column 2 by 13 lines.
      * Reduce page 2, column 1 by 13 lines.
      * Reduce page 2, column 2 by 13 lines.
      * Reduce page 4, column 1 by 13 lines.
      * Reduce page 4, column 2 by 13 lines.
      * Reduce page 5, column 1 by 13 lines.
      * Reduce page 5, column 2 by 13 lines.
      * Reduce page 6, column 1 by 13 lines.
      * Reduce page 6, column 2 by 13 lines.
      * Reduce page 7, column 1 by 13 lines.
      * Reduce page 7, column 2 by 13 lines.
      * Reduce page 8, column 1 by 13 lines.
      * Reduce page 8, column 2 by 13 lines.
      * Reduce page 9, column 1 by 12 lines.
      * Reduce page 9, column 2 by 12 lines.
      * Reduce page 10, column 1 by 12 lines.

    Note: 12 lines = 2.4" = 6.1 cm = 26.8% of the column height

If you're curious how the document managed to shrink substantially as
the result of a relatively minor text reduction, you can typeset the
document with a particular page and column enlarged by a given amount:

    wheretotrim --debug=9,1,12 latex myfile.tex

=head1 CAVEATS

B<wheretotrim> hooks into LaTeX's output routines, which are
notoriously arcane and somewhat fragile.  As a result, it is quite
likely that B<wheretotrim> will fail to analyze a large set of
documents.  Use the B<--verbose> flag to help identify any problems
that B<latex> encounters.

In many cases, B<wheretotrim> will recover by simply ignoring a few
possible page and column expansions.  For example, certain expansions
may result in a C<L<Float(s)
lost|http://www.tex.ac.uk/cgi-bin/texfaq2html?label=fllost>> message.
In other cases, B<wheretotrim> will fail to analyze any modification
to the document.  For example, it may receive an C<Infinite glue
shrinkage found in box being split> error from every page and column
variation it tries.  In this particular case, see the discussion at
L<http://www.michaelshell.org/tex/ieeetran/>.

When B<wheretotrim> is used with a B<latex> auto-build script you may
need to take measures to force the script to rebuild the document even
if it appears that no files have changed.  For example, B<latexmk>
should be given the B<-CF> option to force rebuilding:

    wheretotrim latexmk -CF myfile.tex

=head1 RESTRICTIONS

B<wheretotrim> is implemented as a Perl script with an auxiliary LaTeX
package.  It has been tested only on Linux, but I suspect that it
should also work on S<OS X>.  I doubt it will work under Windows,
though, due to the way the script uses a B<bash>-specific technique
for redirecting the standard error device into the standard output
device.

=head1 AUTHOR

Scott Pakin, I<scott+wtt@pakin.org>

=head1 COPYRIGHT AND LICENSE

Copyright (C) 2013, Scott Pakin <scott+wtt@pakin.org>

This file may be distributed and/or modified under the conditions of
the LaTeX Project Public License, either version 1.3c of this license
or (at your option) any later version.  The latest version of this
license is in:

=over 4

=item E<nbsp>

L<http://www.latex-project.org/lppl.txt>

=back

and version 1.3c or later is part of all distributions of LaTeX
version 2008/05/04 or later.

=head1 SEE ALSO

latex(1),
L<the savetrees package|http://www.ctan.org/pkg/savetrees/>
