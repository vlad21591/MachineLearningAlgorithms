**Machine Learning Algorithms**

**Introduction**:

The sinking of the Titanic is one of the most infamous shipwrecks in history.
On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.
While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

**Purpose**:

Build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).
Machine learning with feature scaling (standardization) to create a model that predicts which passengers survived the Titanic shipwreck.

[The data came from Kaggle -- Titanic: Machine Learning from Disaster]

**Results(from the Conclusion.py)**:

LR(Logistic regression) -- Accuracy: 0.691, Precision: 0.75, Recall:0.231, Latency: 0.01558065414428711ms

SVM(Support Vector Machines) -- Accuracy: 0.635, Precision: 0.0, Recall:0.0, Latency: 0.0ms

MLP(Multilayer perception) -- Accuracy: 0.382, Precision: 0.371, Recall:1, Latency: 0.0ms

RF(Random forest) -- Accuracy: 0.742, Precision: 0.744, Recall:0.446, Latency: 0.0ms

GB(Gradient boosted trees) -- Accuracy: 0.433, Precision: 0.373, Recall:0.815, Latency: 0.0079889297485351

**Conclusion**:

We can see that from the 5 ML models the Random-Forest (learning_rate=0.01,max_depth=2, n_estimators=500) is the best fit. 

Random Forest(with test_data) -- Accuracy: 0.804, Precision: 0.836, Recall:0.671, Latency: 0.046ms

Thank you.
