## Iris and breast cancer datasets
### Iris Dataset 
##### Overview of data
150 samples from these Iris flower species: Setosa, versicolor, and virginica. The features:
- Sepal Length
- Sepal Width
- Petal length
- Petal width
We are tryin to predict the species off of the features
**[Overview + Eda](https://www.loom.com/share/adf6d8e5a53743158a5bb1afd1b33c3d)**
**[plotly docs](https://plotly.com/python/)**
**[Model Prep](https://www.loom.com/share/6e5fc83f4d614e7797f541486c12944b)**
### Importance of standard scaler
KNN relies heavily on distance calcs, this means that the scale of features can significantly impact model performance (if the features vary a lot in scale, larger values will dominate distance calculations, leading to skewed results)
**EX:**
- petal length can be in cm ranging from 1 to 7, but sepal length might have a range of 0.1 to 2, so the petal length would dominate
##### What is standard scaler?
Popular method of preprocessing that standardizes the featurres by removing the mean and scaling unit variance, so all features will follow standard normal distribution and have the same scale
- you use it when:
    - you have features with different units and scales
    - when using distance based algorithims
    - you wanna make sure each feature contributes equally to the model
![image on standard scale](https://cdn.disco.co/media/image_31cb6098-493e-40b5-a63c-c49ac5c02c40.jpeg)
**[scaling the features](https://www.loom.com/share/628ecf1db05c44f58b5b93a061aae383)**
### KNN Key parameters
1. `n_neighbors` (k): Number of neighbors to look at for classification or regression
- default = 5
- smaller k can lead to a more flexible model, but it can cause overfitting if too large
- larger k makes smoother decision boundaries, but can lead to underfitting
```python
knn = KNeighborsClassifier(n_neighbors=3)
```
2. `weights`: determines if the neighbors have equal weight or if closer neighbors have more influence
- `default = 'uniform'` (they will have equal weight)
- `distance` (closer points have more influence)
- the value can also be a custom function
```python
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
# gives more weight to the closer points, useful for noisy datasets
``` 
3. `metric`: the distance measure used to calculate closeness
- default: `mikowski` with `p=2` it is `euclidean`
    - `euclidian` straight line distance
    - `manhattan` sum of absolute differences (AKA L1 distance)
    - `chebyshev` maximum coordinate difference between two points
    - `minkowski` generalization of euclidian and manhattan (requires `p` parameterto define specific distance to use)
- chose matters
    - euclidean for continous values
    - manhattan for categorical values
```python
knn = KNeighborsClassifier(n_neighbors=5, metric='manhattan')
# using manhattan for distance calculation
```
**[KNN for model evaluation](https://www.loom.com/share/841671315e14441a825e87b3580b33db)**
### Breast cancer dataset
dataset has 569 rows, and 30 different columns, like:
- Mean radius: average radius of a tumor
- Mean texture: Variation in texture 
- mean smoothness: the smothness of the tumors edges
- mean symmetry: how symmetrical the tumor is
the target variable is the diagnosis, wether it's benign (0) or malignent (1)
**[Overview + eda](https://www.loom.com/share/44ae202d794941218a128b25a362cd6b)**
**[Basline model + feature selection](https://www.loom.com/share/094edb9e737f41d695564fed64b9600a)**
**[instantiating, fitting, and evaluating the model](https://www.loom.com/share/1a77b013450240378274506dce789e4d)**
**[Exploring KNN to improve model accuracy](https://www.loom.com/share/5d2d8c91850745b88963caee53b76d9e)**
##### Other common classification models
1. Logistiical regression: statisticval model estimates the probability of an outcome based on one ore more predictor values, simple and interpretable, used as a bseline for classification tasks
- **[Logistic Regression Doc](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)**
2. Decisions Tree: tree regressor, splits data based on feature values, and makes decisions based on the majority class in leaf nodes
- **[Decission Tree Doc](https://scikit-learn.org/stable/modules/tree.html#decision-trees)**
3. Random Forest: randomly splits data into several decission trees and averages the results between the trees, works well for regression and classification
- **[Random Forest Doc](https://scikit-learn.org/stable/modules/ensemble.html#random-forests)**
4. Support Vector Machines (SVM): Classifier that finds the hyperplane to best seperate the classes in the feature space, effective in high-dimensional spaces
- **[SVM Doc](https://scikit-learn.org/stable/modules/svm.html#svm)**
5. Naive Bayes: probabilistic classifier based on bayes' theorem. assumes that the features are conditionally independent, fast and effective for text classification tasks
- **[Naive Bayes Doc](https://scikit-learn.org/stable/modules/naive_bayes.html#naive-bayes)**
6. Neural Networks: complex model, inspired by the human brain. composed of layers of inner connected nodes that process input data and predict an output. usseful for large datasets and complex patterns
- **[Neural Network Doc](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#neural-networks-supervised)**
**[Different ML models video](https://www.loom.com/share/4e237f8e063e4eebabfaa33f3eee83ae)**