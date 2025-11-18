## K-Nearest Neighbors
(or KNN) is a simple yet effective ML Algorithim
- theory of KNN
- applications + implementation
- Understand how KNN works
- how to fine tune KNN
- How to evaluate the performance of a KNN model
### A Classification problem
![image one on Problem](https://cdn.disco.co/media/image_ec2e71da-2aa9-4037-963e-ec77f7600df4.jpeg)
![image 2 on Problem](https://cdn.disco.co/media/image__1__cbb4698f-20e8-4624-a91b-2102e2d249c8.jpeg)
![Imaage 3 on Problem](https://cdn.disco.co/media/image__2__af0e1808-e890-49dc-9b4c-5578bb025b80.jpeg)
![Image 4 on Problem](https://cdn.disco.co/media/image__3__702ca9c3-2d0f-4a13-a664-d02bea9c72c9.jpeg)
**Which is the idea behind KNN**
### Understanding K-Nearest Neighbors
KNN is a simple but poweful algorithim used for classification and regression.
- It is a supervised learning algorithim, meaning it needs labeled data to train it.
- Known as a lazy learner, because it doesn't train the model
- it stores the entire training dataset and makes predictions based on the similarity of new inputs compared to the stored data
![Image on KNN](https://cdn.disco.co/media/image__4__d7c62538-a834-4270-be48-d6036efacf6b.jpeg)
### Concepts of KNN
##### What is KNN
- operates on the idea that similar data points will likely have similar outcomes, it predicts the value by:
1. Looks at the nearest neighbors(adjacent points) of a point
2. (in Classification): Assigns the most common classamong the nearest neighbors
3. (in Regression): Calculates the average value of the neighbors, and uses that for the point
##### Point of "K" in KNN
- parameter of KNN
    - `K = 1` would look at just the closest point value to the point we are trying to predict
    - `K = a number bigger than 1` would look at mopre points, but to many points would lead to the algorithim become to generalized
- Tip: The optimal K value is usually determined through techniques, like splitting into training and testing sets. It is about balancing noise sensitivity with noise generalization
##### Distance metrics in KNN
- distance metrics are what find the nearest neighbors, the most commonly used ones:
    - Euclidean Distance (common + the default)
![Image of euclidean distance](https://cdn.disco.co/media/image__5__b5435568-f515-4209-82c6-195bb72abffb.jpeg)
    - Manhattan Distance
![Image of manhattan distance](https://cdn.disco.co/media/image__6__df5efaef-0076-40b8-95d2-08d050887aaf.jpeg)
    - Minkowski distance (generalized form)
![Image of Minkowski form](https://cdn.disco.co/media/image__7__2be14f10-852c-464e-b672-ccc810bb1431.jpeg)
    - Hamming distance (for categorical values)
- *Important*: Always scale your features before using KNN, features with larger ranges will dominate distance calculations
##### Voting mechanics in KNN
- each K neighbor will vote on a points class, the class with the most votes wins, it uses two types of voting:
    - majority voting: Each neighbor has equal weight (democracy)
    - weighted voting: Closer neighbors have more influence (electoral college)
- *Tip:* for weighted voting, you can use the inverses distance as the weight to ensure closer neighbors have more impact
### Steps in KNN
1. Chose the Value of K: select the number of nearest neighbors (k) you want to use to make predictions, for the problem below, it is a k value of 5
2. Calculate distance: for each new point (orange poit as below), you calculate the distance between that point and all the other points in the dataset using a distance metric (shorter distance means the points are more similar)
3. find K Nearest Neighbors: sort the distances and select the nearest neighbors. in the image the nearest neighbors are circled (the category doesn't matter)
4. Voting for classification: if the task is classification: each neighbor voted for its class. the point is assigned to the class with the most votes (weighted or majority)
5. prediction: based on the majority vote, assign the new point to a predicted class. in this cass it would be category a (if it is majotiy, though wweighted could make a difference?)
![Image of KNN "In Action"](https://cdn.disco.co/media/image__8__d3cb1898-c05e-4491-8c4c-db0888ce8e0e.jpeg)
### Advantages + Disadvantages
##### Advantages
- Simple: easy to interpret and use
- No Training phase needed: No explicit training process is needed
- Flexible: used for both classification and regression
- adaptable: can handle complex, non-linear decision bounderies
##### Disadvatages
- Computationally expensive: As it has to calculate the distance between each new point, and compares each of those new points to every other point in the dataset to find the nearest neighbors
- Sensitive to noise: espicially when K is small
- requires features scaling: Different feature sclaes can skew the distance metrics