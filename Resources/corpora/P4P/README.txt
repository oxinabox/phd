//////////////////
//              //
//  P4P Corpus  //
//              //
//////////////////


Overview
Folder Contents
Format
Tagset
Note on Terminology
Referencing
Acknowledgements
Contact
Last Revision


[Overview]

P4P stands for "Paraphrase for Plagiarism". The P4P corpus is composed of a partition of the 
plagiarism cases in the PAN-PC-10 corpus [2]. It is composed of 847 source-plagiarism pairs 
manually annotated with the paraphrase phenomena they contain.  

The tagset consists of 20 paraphrase types plus identical and non-paraphrase cases (see [Tagset]).

The P4P corpus is freely available for research purposes at 

	http://clic.ub.edu/corpus/en/paraphrases-download-en (package to download)
        http://clic.ub.edu/corpus/en/paraphrase_search (search interface)

Annotation guidelines are available at 

        http://clic.ub.edu/corpus/en/paraphrases-en 
		
For further reading on the corpus, refer to [1]

[1] A. Barrón-Cedeño, M. Vila, M.A. Martí, and P. Rosso. 2013. Plagiarism meets paraphrasing: Insights 
    for the next generation in automatic plagiarism detection. Computational Linguistics. 
    To appear in issue 39:4. DOI: 10.1162/COLI_a_00153 
    (http://www.mitpressjournals.org/doi/pdf/10.1162/COLI_a_00153).

[2] M. Potthast, B. Stein, A. Barrón-Cedeño, and P. Rosso. 2010. An evaluation framework for plagiarism 
    detection. In COLING 2010 Proceedings of the 23rd International Conference on Computational 
    Linguistics, pages 997-1005, Beijing, China, August 2010 
    (www.aclweb.org/anthology-new/C/C10/C10-2115.pdf).


[Folder Contents]

- The P4P corpus                                P4P_corpus_v1.xml
- Sample XML file from the PAN-PC-10 corpus     suspicious-document00061.xml
- This readme file                              README.txt


[Format]

In order to set out the corpus format, we use the following example:

[1] <relacio>
[2]   <id>9249</id>
[3]     <frase fnum="1" type="plagiarism" plagiarism_reference="00061" offset="47727" length="182">
[4]		All art is imitation of nature. One does not need to recognize a tangible object to be moved by its artistic representation. 
[5]		Here by virtue of humanity's vestures, lies its appeal.
[6]     </frase>
[7]     <frase fnum="2" type="source" source_reference="source-document05510.txt" offset="98424" length="249">
[8]		In order to move us, it needs no reference to any recognised original. It is there in virtue of the vesture of humanity 
[9]		in which it is clothed, and makes its appeal at once and directly.  It is usual to speak of all the fine arts as imitative arts.
[10]    </frase>
[11]    <anotacio es_parafrasi="1" autor="87">
[12]      <id>18689</id>
	[...]
[13]      <fenomen type="lex_same_polarity" scope="local">
[14]        <wordrange fnum="1" wdranges="15" offset="47801" localoffset="74" length="6" offsets="47801" localoffsets="74" lengths="6">
[15]          <fragment>object</fragment>
[16]        </wordrange>
[17]        <wordrange fnum="2" wdranges="13" offset="98485" localoffset="61" length="8" offsets="98485" localoffsets="61" lengths="8">
[18]          <fragment>original</fragment>
[19]        </wordrange>
	[...]
[20]      <fenomen type="syn_negation">
[21]        <wordrange fnum="1" wdranges="7-23" offset="47759" localoffset="32" length="92" offsets="47759" localoffsets="32" lengths="92">
[22]          <fragment>One does not need to recognize a tangible object to be moved by its artistic representation.</fragment>
[23]          <focus ranges="9" offset="47768" localoffset="41" length="3" offsets="47768" localoffsets="41" lengths="3">not</focus>
[24]        </wordrange>
[25]        <wordrange fnum="2" wdranges="0-14" offset="98424" localoffset="0" length="70" offsets="98424" localoffsets="0" lengths="70">
[26]          <fragment>In order to move us, it needs no reference to any recognised original.</fragment>
[27]          <focus ranges="8" offset="98454" localoffset="30" length="2" offsets="98454" localoffsets="30" lengths="2">no</focus>
[28]        </wordrange>
[29]      </fenomen>
	[...]
[30]    </anotacio>
[31] </relacio>


Each tag "relacio" (line [1]) corresponds to a source-plagiarism pair annotation. In lines [3-10] 
two phrases are included: one of type plagiarism and one of type source. The metadata in the tags 
"frase" correspond to the original information from the PAN-PC-10 corpus (the PAN-PC-10 XML file 
suspicious-document00061.xml is included in this zip file for explanatory purposes). Considering 
lines [3] and [7] in our example, the information can be easily mapped to that of the PAN-PC-10 
corpus as follows:

+-----------------------------------------------+--------------------------------------------------+
|                     P4P                       |      PAN-PC-10 (suspicious-document00061.xml)    |
+-----------------------------------------------+--------------------------------------------------+
|fnum="1" type="plagiarism"                     |                                                  |
|   plagiarism_reference="00061"                | document reference="suspicious-document00061.txt |
|   offset="47727"                              | this_offset="47727"                              |
|   length="182"                                | this_length="182"                                |
|                                               |                                                  |
|fnum="2" type="source"                         |                                                  |
|   source_reference="source-document05510.txt" | source_reference="source-document05510.txt"      |
|   offset="98424"                              | source_offset="98424"                            |
|   length="249"                                | source_length="249"                              |
+-----------------------------------------------+--------------------------------------------------+	
		
Each paraphrase phenomenon is labelled with the tag "fenomen". In tags corresponding to morphology, 
lexicon and semantics based classes, and order and addition_deletion tags, this XML tag includes two 
parameters: "type" and "scope". In the rest of tags, "fenomen" only includes the parameter "type". 
Parameter "type" corresponds to the type of paraphrase in our typology and "scope" corresponds to the 
impact of the phenomenon in the rest of the text fragment. "Scope" values may be "local", meaning 
there is an impact, or "global", meaning there is not. For instance, the phenomenon starting in line 
[13] corresponds to the type "same polarity substitution" and its scope is "local". 

Every "fenomen" contains two "wordrange" tags. In our example, in lines [14-19]. The first wordrange 
tag (lines [14-16]) corresponds to the "plagiarism" fragment, as both include fnum="1". The tag in 
lines [17-19] corresponds to the source fragment. The offset and length are given at character level 
and, again, can be mapped to the PAN-PC-10 corpus. The addition/deletion type only contains a 
"wordrange" tag corresponding either to the "plagiarism" fragment or to the "source". Inside of 
wordrange, we have the actual text fragments (lines [15] and [18]). In this case, the source fragment 
was "object" and, after performing the same-polarity substitution, the text in the plagiarised 
fragment became "original". In tags corresponding to syntax and discourse classes, and the 
syn_dis_structure tag, a "focus" tag is generally provided, containing the most relevant snippet in 
"fragment". 	

Lines [30] and [31] close this source-plagiarism case.


[Tagset]

The different tags correspond to 7 classes:

+---------------------------+---------------------------+-----------------------------------+
|          Class            |           Tag             |             Meaning               |
+---------------------------+---------------------------+-----------------------------------+
|                           | mor_inflectional          | inflectional changes              |
| Morphology based changes  | mor_modal_verb            | modal verb changes                |
|                           | mor_derivational          | derivational changes              |
+---------------------------+---------------------------+-----------------------------------+
|                           | lex_spelling_and_format   | spelling and format changes       |
|                           | lex_same_polarity         | same polarity substitutions       |
| Lexicon based changes     | lex_synt_ana              | synthetic/analytic substitutions  |	
|                           | lex_opposite_polarity     | opposite polarity substitutions   |
|                           | lex_converse              | converse substitutions            |
+---------------------------+---------------------------+-----------------------------------+
|                           | syn_diathesis             | diathesis alternations            |
|                           | syn_negation              | negation switching                |
| Syntax based changes      | syn_ellipsis              | ellipsis                          |
|                           | syn_coordination          | coordination changes              |
|                           | syn_subord_nesting        | subordination and nesting changes |
+---------------------------+---------------------------+-----------------------------------+
|                           | dis_punct_format          | punctuation and format changes    |
| Discourse based changes   | dis_direct_indirect       | direct/indirect style alternations|
|                           | dis_sent_modality         | sentence modality changes         |
|                           | syn_dis_structure         | syntax/discourse structure changes|
+---------------------------+---------------------------+-----------------------------------+
| Semantics based changes   | semantic                  | semantic based changes            |
+---------------------------+---------------------------+-----------------------------------+
| Miscellaneous changes     | order                     | change of order                   |
|                           | addition_deletion         | addition/deletion                 |
+---------------------------+---------------------------+-----------------------------------+
| Others                    | identical                 | identical                         |
|                           | non_paraphrases           | non-paraphrases                   |
+---------------------------+---------------------------+-----------------------------------+  
 

[Note on Terminology]

Sometimes the terms used in the corpus and in the paper in [1] differ. In the following, a 
mapping between the two is provided:

PAPER --> CORPUS
scope --> range 
projection (local/global) --> scope (local/global)
key element --> focus
 

[Referencing]

Please cite the following paper when using the corpus

@article{Barron+etal:13,
 AUTHOR = "Barr{\'o}n-Cede{\~n}o, Alberto and 
	   Vila, Marta and 
	   Mart{\'i}, {M. Ant{\`o}nia} and 
	   Rosso, Paolo",
Title = "Plagiarism Meets Paraphrasing: Insights for the Next Generation in Automatic Plagiarism Detection",
Journal = "Computational Linguistics",
Volume = "39",
Number = "4",
Note    = "{DOI}: 10.1162/COLI\_a\_00153", 
Year = "2013, to appear"
}


[Acknowledgements]

We would like to thank the people that participated in the annotation of the P4P corpus.
This research work was partially carried out during the tenure of an ERCIM "Alain Bensoussan" 
Fellowship Programme. The research leading to these results received funding from the EU FP7 
Programme 2007-2013 (grant n. 246016), the MICINN projects TEXT-ENTERPRISE 2.0 and 
TEXT-KNOWLEDGE 2.0 (TIN2009-13391), the EC WIQ-EI IRSES project (grant n. 269180), and the 
FP7 Marie Curie People Programme. The research work of A. Barrón-Cedeño and M. Vila was financed 
by the CONACyT-Mexico 192021 grant and the MECD-Spain FPU AP2008-02185 grant, respectively.
The research work of A. Barrón-Cedeño was partially done in the framework of his PhD at the 
Universitat Politècnica de València.



[Contact]

We would like to know what the P4P corpus is useful for.
For comments and other issues, refer to:

	http://clic.ub.edu/en/users/marta-vila-rigat	 

	http://www.lsi.upc.edu/~albarron		 		
	

[Last Revision]

January 2013
