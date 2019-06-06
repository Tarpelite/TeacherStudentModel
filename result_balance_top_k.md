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
|128             |4.0               |4.0              |10.0                  |3.0                  |**50**|


#### results
|    Method                     |  trec6        | dbpedia-10k |   yelp-10k|
|  :---------                   | :------:      | :------:    | :------:  |
|teacher model/original bert    | 0.704019      | 0.886667    | 0.396263  |
|student model before ft        | 0.614373      | 0.846667    | 0.355253  |
|student model after ft         | 0.737174      | 0.857374    | 0.402424  | 

### Task 2

#### settings
| max_seq_length | train_batch_size | eval_batch_size | teacher_train_epochs | student_train_epochs|top_k|
|:---------------|:-----------------|:----------------|:---------------------|:--------------------|:----|
|128             |4.0               |4.0              |10.0                  |3.0                  |**200**|


#### results
|    Method                     |  trec6        | dbpedia-10k |   yelp-10k|
|  :---------                   | :------:      | :------:    | :------:  |
|teacher model/original bert    | 0.704019      | 0.886667    | 0.396263  |
|student model before ft        | 0.658820      | 0.872121    | 0.388788  |
|student model after ft         | 0.740878      | 0.888687    | 0.414242  | 

### Task 3

#### settings
| max_seq_length | train_batch_size | eval_batch_size | teacher_train_epochs | student_train_epochs|top_k|
|:---------------|:-----------------|:----------------|:---------------------|:--------------------|:----|
|128             |4.0               |4.0              |10.0                  |3.0                  |**500**|


#### results
|    Method                     |  trec6        | dbpedia-10k |   yelp-10k|
|  :---------                   | :------:      | :------:    | :------:  |
|teacher model/original bert    | 0.704019      | 0.886667    | 0.396263  |
|student model before ft        | 0.749583      | 0.908384    | 0.392727  |
|student model after ft         | 0.771995      | 0.898283    | 0.439293  | 

