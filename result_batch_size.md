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
|128             |4.0               |4.0              |10.0                  |3.0                  |200|


#### results
|    Method                     |  trec6        | dbpedia-10k |   yelp-10k|
|  :---------                   | :------:      | :------:    | :------:  |
|teacher model/original bert    | 0.704019      | 0.886667    | 0.396263  |
|student model before ft        | 0.651232      | 0.872020    | 0.369495  |
|student model after ft         | 0.777181      | 0.897475    | 0.410909  | 

### Task 2

#### settings
| max_seq_length | train_batch_size | eval_batch_size | teacher_train_epochs | student_train_epochs|top_k|
|:---------------|:-----------------|:----------------|:---------------------|:--------------------|:----|
|128             |2.0               |2.0              |10.0                  |3.0                  |200|


#### results
|    Method                     |  trec6        | dbpedia-10k |   yelp-10k|
|  :---------                   | :------:      | :------:    | :------:  |
|teacher model/original bert    | 0.693277      | 0.863434    | 0.400000  |
|student model before ft        | 0.731432      | 0.876262    | 0.367980  |
|student model after ft         | 0.769772      | 0.877778    | 0.423131  | 

### Task 3

#### settings
| max_seq_length | train_batch_size | eval_batch_size | teacher_train_epochs | student_train_epochs|top_k|
|:---------------|:-----------------|:----------------|:---------------------|:--------------------|:----|
|128             |1.0               |1.0              |10.0                  |3.0                  |200|


#### results
|    Method                     |  trec6        | dbpedia-10k |   yelp-10k|
|  :---------                   | :------:      | :------:    | :------:  |
|teacher model/original bert    | 0.736247      | 0.868485    | 0.448990  |
|student model before ft        | 0.733099      | 0.879596    | 0.409293  |
|student model after ft         | 0.748472      | 0.860606    | 0.411515  | 

