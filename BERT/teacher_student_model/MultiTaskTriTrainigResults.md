# MultiTaskTriTrainigResults

## Sentiment Analysis

## Dataset

| Domain | labeled | unlabeled | dev | tests|
| ---    | ---     | ---       | --- | ---  | 
| Book   | 2000    | 4465      | 200 | 6000 |
| DVD    | 2000    | 3586      | 200 | 6000 |
| Electronics| 2000| 5681      | 200 | 6000 |
| Kitchen| 2000    | 5945      | 200 | 6000 |

## Results


|Model        | D   | B   | E   | K   | Avg  |
|-            | -   | -   | -   | -   | -    |
|origin(CNN)  |75.91|73.47|75.61|79.58|76.14 |
|MT-tri(CNN)  |78.14|74.86|81.45|82.14|79.15 |
|origin(BERT) |88.00|91.62|93.18|93.67|91.62 |
|MT-tri(BERT) |90.10|91.48|92.85|94.61|92.09 |

