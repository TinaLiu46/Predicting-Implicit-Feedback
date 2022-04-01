# PREDICTING IMPLICIT RATINGS
## Table of Contents
1. [Description of the problem](#description-of-the-problem)
2. [Negative Sampling](#negative-sampling)
3. [Models](#models)
   1. [Matrix Factorization](#matrix-factorization)
   2. [Neural Network](#neural-network)
5. [Conclusion & Lesson Learned](#conclusion-and-lesson-learned)

## 📜Description of the problem
   Given userid,itemid,contextfeatureid, and itemfeatureid, the goal is to predict probabilities of users' preference of certain item. The example of dataset is shown below:
   | user_id  | item_id | context_feature_id  | item_feature_id |
| ------------- | ------------- | ------------- | ------------- |
| 23542  | 46423 | 32 | 3 |
## 📑Negative Sampling
   - Uniform Random Sampling
   - User-Oriented Sampling: a user has viewed more items, those items that she has not viewed be sampled with higher probability
   - Item-Oriented Sampling: inverse relation to popularity
   - __Hybrid(Our Method)__:
      - Step1: Assign weight to each user based on their rating frequency
      - Step2: Assign weight to each item based on their inverse popularity (softmax the inverse)
     
      Note*: Cold_user and Cold_item problem:
         
       - 'Cold user' we gave a fixed amount of negative samples
       - 'Cold item' we gave the same weight as itmes who have frequency of 1
 ```python
 def calculate_weight(df,col,softmax_ = None,cold_item = None):
    total = len(df)
    item_count = df.groupby([col]).size().values
    if softmax_ == None:
        if cold_item == None:
            item_weight = item_count/total
        else:
            item_count_cold = list(np.array(item_count)) + list([1]*len(cold_item))
            item_weight_ = np.array(item_count_cold)/(total+len(cold_item))
            cold_item_weight = np.random.dirichlet(np.ones(len(cold_item)))
            a = 0.8
            item_weight = list(item_weight_ * a) + list(cold_item_weight*(1-a))
    else:
        if cold_item == None:
            item_weight_reversed = 1/(item_count/total)
            item_weight = softmax(item_weight_reversed)
        else:
            item_count_cold = list(np.array(item_count) + 1) + list([1]*len(cold_item))
            item_weight_reversed_cold = 1/(np.array(item_count_cold)/(total+len(cold_item)))
            item_weight = softmax(item_weight_reversed_cold)
    return item_weight
 ```
      
      
## 📈Models

### Matrix Factorization
Matrix factorization is a class of collaborative filtering algorithms used in recommender systems. Matrix factorization algorithms work by decomposing the user-item interaction matrix into the product of two lower dimensionality rectangular matrices.
```python
class MF(nn.Module):
    def __init__(self, num_users, num_items, emb_size=100):
        super(MF, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)  # .cuda()
        self.item_emb = nn.Embedding(num_items, emb_size)  # .cuda()
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        # initlializing weights
        self.user_emb.weight.data.uniform_(0, 0.05)  # .cuda()
        self.item_emb.weight.data.uniform_(0, 0.05)  # .cuda()
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_bias.weight.data.uniform_(-0.01, 0.01)

    def forward(self, u, v):
        U = self.user_emb(u)  # .cuda()
        V = self.item_emb(v)  # .cuda()

        b = self.user_bias(u).squeeze()
        c = self.item_bias(v).squeeze()
        # return (U*V).sum(1)
        return (U*V).sum(1)+b+c
  ```
- Hyperparameter Tuning(Best):

         Emb_size: 1000
         Epochs: 15
         Learning Rate: 0.001
- Metrics(Best):

         Train Loss: 0.218
         Valid Loss: 0.228
         Test Y_hat Mean: 0.621

### Neural Network
By applying nueral network, we added one linear layer and used ReLU as the activation layer.

```python
class CollabFNet(nn.Module):
    def __init__(self, num_users, num_items,num_item_fea, num_context_fea,emb_size=100, n_hidden=10):
        super(CollabFNet, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)
        self.item_fea_emb = nn.Embedding(num_item_fea, emb_size)
        self.context_fea_emb = nn.Embedding(num_context_fea, emb_size)       
        self.lin1 = nn.Linear(emb_size*4, n_hidden)
        self.lin2 = nn.Linear(n_hidden, 1)
        self.drop1 = nn.Dropout(0.1)
        
    def forward(self, u, v,a,b):
        
        U = self.user_emb(u)
        V = self.item_emb(v)
        A = self.item_fea_emb(a)
        B = self.context_fea_emb(b)
        
        x = F.relu(torch.cat([U, V,A,B], dim=1))
        x = self.drop1(x)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
 ```
 - Hyperparameter Tuning(Best):

         Emb_size: 50
         Epochs: 20
         Learning Rate: 0.05
         Weight Decay: 1e-6
         Dropout: 0.1
         Hidden Layer_size: 10
- Metrics(Best):

         Train Loss: 0.192 
         Valid Loss: 0.210
         Test Y_hat Mean: 0.625


## 💡Conclusion and Lesson Learned
- Negative Sampling - Cold start (both user and item) is a very challenging issue in real world redommandation system. Negative sampling cold starters is essential.
- Write things in Function or Class format - can be reused easily
- Understand the pro/cons of different ML models – use the most
efficient and effective one
- Hyperparameter Tuning is Very Important – thoroughly search parameters

## References:
- https://medium.com/analytics-vidhya/recommender-systems-explicit-feedback-implicit-feedback-and-hybrid-feedback-ddd1b2cdb3b
- https://en.wikipedia.org/wiki/Matrix_factorization_(recommender_systems)
