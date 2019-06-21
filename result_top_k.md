# Experiments(July 6) 

## Data
|    Dataset    |  #Points  |  # Classes | % of largest class |
|  :---------   | :------:  | :------:   | :------:           |
|  TREC-6       | 5452      | 6          | 22.9%              |
|  dbpedia-10k  | 10000     | 14         | 7.15%              |
|  yelp-10k     | 10000     | 5          | 20.03%             |

## Classification Results (1% as trainig and the rest as test)

### Task 1

#### settings
| max_seq_length | train_batch_size | eval_batch_size | teacher_train_epochs | student_train_epochs|top_k|
|:---------------|:-----------------|:----------------|:---------------------|:--------------------|:----|
|128             |4.0               |4.0              |10.0                  |3.0                  |**200**|


#### results
|    Method                     |  trec6        | dbpedia-10k |   yelp-10k|
|  :---------                   | :------:      | :------:    | :------:   |
|teacher model/original bert    | 0.704019      | 0.886667    | 0.396263   |
|student model before ft        | 0.651232      | 0.872020    | 0.369495   |
|student model after ft         | **0.777181**  | 0.897475    |**0.410909**| 

### Task 2

#### settings
| max_seq_length | train_batch_size | eval_batch_size | teacher_train_epochs | student_train_epochs|top_k|
|:---------------|:-----------------|:----------------|:---------------------|:--------------------|:----|
|128             |4.0               |4.0              |10.0                  |3.0                  |**300**|


#### results
|    Method                     |  trec6        | dbpedia-10k |   yelp-10k|
|  :---------                   | :------:      | :------:    | :------:  |
|teacher model/original bert    | 0.704019      | 0.886667    | 0.396263  |
|student model before ft        | 0.667531      | 0.897273    | 0.384646  |
|student model after ft         | 0.760696      | 0.858889    | 0.381616  | 

### Task 3

#### settings
| max_seq_length | train_batch_size | eval_batch_size | teacher_train_epochs | student_train_epochs|top_k|
|:---------------|:-----------------|:----------------|:---------------------|:--------------------|:----|
|128             |4.0               |4.0              |10.0                  |3.0                  |**500**|


#### results
|    Method                     |  trec6        | dbpedia-10k |   yelp-10k|
|  :---------                   | :------:      | :------:    | :------:  |
|teacher model/original bert    | 0.704019      | 0.886667    | 0.396263  |
|student model before ft        | 0.750880      | **0.901414**    | 0.404444  |
|student model after ft         | 0.726616      | 0.899495    | 0.387879  | 

### Task 4

#### settings
| max_seq_length | train_batch_size | eval_batch_size | teacher_train_epochs | student_train_epochs|top_k|
|:---------------|:-----------------|:----------------|:---------------------|:--------------------|:----|
|128             |4.0               |4.0              |10.0                  |3.0                  |**1000**|


#### results
|    Method                     |  trec6        | dbpedia-10k |   yelp-10k|
|  :---------                   | :------:      | :------:    | :------:  |
|teacher model/original bert    | 0.704019      | 0.886667    | 0.396263  |
|student model before ft        | 0.705686      | 0.887273    | 0.398384  |
|student model after ft         | 0.708835      | 0.876869    | 0.390909  | 