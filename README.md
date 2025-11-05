Overview

This repository contains the scripts used for text preprocessing and topic modeling using BERTopic.

Files

1. Preprocessing.py – Implements all text preprocessing steps, including:

   a) Removal of annotation tags

   b) Exclusion of greetings within the first ten words

   c) Lemmatization using Stanza

   d) Application of a custom conversion list (included) to correct incorrectly lemmatized words

2. BERTopic pipeline.py – Contains the complete BERTopic workflow, including model initialization, topic generation, and handling of stop words (file included)
3. Count emotion words.py - Counts a list of emotion words or lemmas present in a set of texts (emotion lemma files included)
4. Analyses.py - Counts number of unique emotion words per topic, number of unique topics per emotion word; sums use of each topic and calculates Gini coefficient per participant
