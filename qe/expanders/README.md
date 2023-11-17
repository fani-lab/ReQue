# Expanders
The objective of query refinement is to produce a set of potential candidate queries that can function as enhanced and improved versions. This involves systematically applying various unsupervised query refinement techniques to each query within the input dataset.

<table align="center" border=0>
<thead>
  <tr><td colspan="3" style="background-color: white;"><img src="./classdiagram.png", width="1000", alt="ReQue: Class Diagram"></td></tr>     
  <tr><td colspan="3">
      <p align="center">Class Diagram for Query Expanders in <a href="./qe">qe/</a>. [<a href="https://app.lucidchart.com/documents/view/64fedbb0-b385-4696-9adc-b89bc06e84ba/HWEp-vi-RSFO">zoom in!</a>].</p>
      <p align="center"> The expanders are initialized by the Expander Factory in <a href="./qe/cmn/expander_factory.py">qe/cmn/expander_factory.py</a></p></td></tr> 
 </thead>
</table>

Here is the list of queries:
| **Expander** 	| **Category** 	| **Analyze type** 	|
|---	|:---:	|:---:	|
| [adaponfields](Adaponfields) 	| Top_Documents 	| Local 	|
| [anchor](Anchor) 	| Anchor_Text 	| Global 	|
| [backtranslation](Backtranslation) 	| Machine_Translation 	| Global 	|
| [bertqe](Bertqe) 	| Top_Documents 	| Local 	|
| [conceptluster](Conceptluster) 	| Concept_Clustering 	| Local 	|
| conceptnet 	| Semantic_Analysis 	| Global 	|
| docluster 	| Document_Summaries 	| Local 	|
| glove 	| Semantic_Analysis 	| Global 	|
| onfields 	| Top_Documents 	| Local 	|
| relevancefeedback 	| Top_Documents 	| Local 	|
| rm3 	| Top_Documents 	| Local 	|
| sensedisambiguation 	| Semantic_Analysis 	| Global 	|
| stem.krovetz 	| Stemming_Analysis 	| Global 	|
| stem.lovins 	| Stemming_Analysis 	| Global 	|
| stem.paicehusk 	| Stemming_Analysis 	| Global 	|
| stem.porter 	| Stemming_Analysis 	| Global 	|
| stem.sstemmer 	| Stemming_Analysis 	| Global 	|
| stem.trunc 	| Stemming_Analysis 	| Global 	|
| tagmee 	| Wikipedia 	| Global 	|
| termluster 	| Term_Clustering 	| Local 	|
| thesaurus 	| Semantic_Analysis 	| Global 	|
| wiki 	| Wikipedia 	| Global 	|
| word2vec 	| Semantic_Analysis 	| Global 	|
| wordnet 	| Semantic_Analysis 	| Global 	|

# Adaponfields
# Anchor

# Backtranslation
Back translation, also known as reverse translation or dual translation, involves translating content, whether it is a query or paragraph, from one language to another and retranslating it to the original language. This method provides several options for the owner to make a decision that makes the most sense based on the task at hand.
For additional details, please refer to this [document](https://docs.google.com/document/d/1K5zPymfH-PfDlJBxqdSHsvMBY7Fb_Dw7WxgBoKFNUIM/edit?usp=sharing).

## Example
| **q** 	| **map q** 	| **language** 	| **translated q** 	| **backtranslated q** 	| **map q'** 	|
|---	|:---:	|:---:	|:---:	|:---:	|:---:	|
| Italian nobel prize winners 	| 0.2282 	| farsi 	| برندهای جایزه نوبل ایتالیایی 	| Italian Nobel laureates 	| 0.5665 	|
| banana paper making 	| 0.1111 	| korean 	| 바나나 종이 제조 	| Manufacture of banana paper 	| 1 	|
| figs 	| 0.0419 	| tamil 	|  அத்திமரங்கள்  	| The fig trees 	| 0.0709 	|

# Bertqe
# Conceptluster
