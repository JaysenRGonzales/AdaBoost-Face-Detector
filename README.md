# Multimethod-Face-Detector
In this project, we attempt to implement a face detector that is trained using AdaBoost, combining information from skin color and rectangle filters, and utilizes the ideas of bootstrapping and classifier cascades. In particular, the following methods/concepts will be utilized in this face detector:

* **AdaBoost**: The face detector, or at least components of the face detector, will be trained using AdaBoost and rectangle filters.
* **Skin detection**: A skin detector must be used to improve the efficiency of face detection on color images.
* **Bootstrapping**: Bootstrapping is a method for improving the quality of the training set, by identifying and including more challenging examples. Bootstrapping is performed by iterating between 1). Training a face detector using the training set, and 2). Applying the face detector on additional data, and adding to the training set cases where the face detector makes mistakes.
* **Classifier cascades**: A classifier cascade is a sequence of classifiers, where the first classifier is very fast but relatively inaccurate, and each subsequent classifier is slower but more accurate. For every classifier in the cascade (except for the final classifier), we need to choose a threshold that determines whether a window should be classified as nonface, or should be passed on to the next classifier in the cascade. That threshold should be chosen so that it causes as few mistakes as possible.

# Authors
Ozy Vielma & Jaysen Gonzales
