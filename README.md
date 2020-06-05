NMF-basis clustering for BSS
====

## Overview
Implementation of a blind source separation (BSS) algorithm using NMF and source-filter based clustering approach.

The "NMF-basis_clustering_for_BSS.py" has been implemented for non-supervised music source separation, using the algorithm proposed by M. Spiertz and V. Gnann [1]. To get well separated basis, the NMF would be optimized with a temporal continuity cost function [2].


## Requirement
soundfile 0.10.3

matplotlib 3.1.0

numpy 1.18.1

scipy 1.4.1

scikit-learn 0.21.2

museval 0.3.0 (only for evaluation metrics)


## Dataset preparation
In the "music" directory, please put 3 music files (.wav format), namely a mixed music, the first instrument play, and the second instrument play. The later two would be used for the evaluation only. For instance, professionally produced music recordings are provided in the SiSEC2011 website [3].


## References
[1] M. Spiertz and V. Gnann: 'Source-Filter Based Clustering for Monaural Blind Source Separation', In Proceedings of the International Conference on Digital Audio Effects, pp.1-4, (2009)

[2] T. Virtanen: 'Monaural Sound Source Separation by Nonnegative Matrix Factorization With Temporal Continuity and Sparseness Criteria', In Proceedings of the IEEE Transactions on Audio, Speech, and Language Processing (TASLP), vol.15, No.3, pp.1066–1074, (2007)

[3] S. Araki, F. Nesta, E. Vincent, Z. Koldovsky, G. Nolte, A. Ziehe, and A. Benichoux: 'The 2011 Signal Separation Evaluation Campaign (SiSEC2011)', [Online], Available: http://sisec2011.wiki.irisa.fr/tiki-index165d.html?page=Professionally+produced+music+recordings, (2011)