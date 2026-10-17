# Data card: Imvo Zabantsundu, language-split chunks

Settings: chunk_words=60, min_words_per_lang=1000, boilerplate = >= 30% of 5-gram shingles seen in >= 5 issues, seed=13

Issues found: 517; kept (>= 1000 words in both languages): 511

| Year | xh chunks | en chunks | xh words | en words | xh boilerplate | en boilerplate |
|---|---|---|---|---|---|---|
| 1884 | 740 | 425 | 44,347 | 24,894 | 25.0% | 1.9% |
| 1885 | 4,561 | 2,650 | 273,676 | 155,309 | 20.7% | 2.2% |
| 1886 | 5,104 | 2,254 | 306,151 | 130,302 | 27.1% | 3.3% |
| 1887 | 5,151 | 2,085 | 309,064 | 122,683 | 28.6% | 5.2% |
| 1888 | 6,323 | 2,324 | 379,101 | 136,962 | 21.1% | 5.3% |
| 1889 | 5,834 | 1,917 | 349,680 | 113,514 | 24.8% | 6.1% |
| 1890 | 6,105 | 2,277 | 365,764 | 134,452 | 28.2% | 7.6% |
| 1891 | 7,692 | 2,877 | 461,193 | 170,549 | 31.8% | 12.3% |
| 1892 | 7,648 | 2,868 | 459,137 | 169,682 | 35.7% | 13.3% |
| 1893 | 8,014 | 2,935 | 480,866 | 173,444 | 36.8% | 13.4% |
| 1894 | 7,980 | 2,620 | 478,891 | 155,471 | 35.8% | 7.3% |

Totals: en: 25,232 chunks, 1,487,262 words, 7.9% boilerplate, 2.3% language mismatch; xh: 65,152 chunks, 3,907,870 words, 29.9% boilerplate, 0.3% language mismatch

Issues per split: train=357, test=77, val=77

Usable chunks (not boilerplate, no language mismatch) per split and language:

| Split | xh | en |
|---|---|---|
| test | 6,680 | 3,609 |
| train | 31,893 | 16,052 |
| val | 6,892 | 3,266 |
