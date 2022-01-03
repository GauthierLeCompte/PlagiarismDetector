# PlagiarismDetector
In src/main.py you will find our implementation for the plagiarism detector.
At the top of that file you will find six global variables. You can change these variables to your needs.

- BANDS: integer of how many bands you use.
- SHINGLES: integer of what de shingle size is.
- SIGNATURE_LENGTH: integer of how long each signature is (the row size).
- TRESHHOLD: float in range 0-1 of the similarity threshold.
- SMALLINPUT: boolean that is true if you use the small dataset and false if you want to use the large dataset.
- RECALCULATE_JACCARD: boolean that is true is you want to calculate the jaccard similarity. It will write a csv with
the similarity in the output directory. If this boolean is false it will read from the previously mentioned csv.

In the output directory you will find multiple files with a name like "...._x_y.png".
The x stands for the shingle size and the y stands for the similarity tresshold.
In the main directory you will find the result of the large dataset.