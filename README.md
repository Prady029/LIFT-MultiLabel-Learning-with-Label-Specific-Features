# LIFT-MultiLabel-Learning-with-Label-Specific-Features

A research paper implementation from scratch for pet project :)

# Why LIFT?

Multi-label learning deals with the problem where each training example is represented by a single instance while associated with a set ofclass labels. For an unseen example, existing approacheschoose to determine the membership of each possible class label to it based onidentical feature set, i.e. the very instance representation of the unseen example is employed in the discrimination processes of all labels. However, this commonly-used strategy might be suboptimal as different class labels usually carry speciﬁc characteristics of their own, and it could be beneﬁcial to exploit different feature sets for the discrimination of different labels. Based on the above reﬂection, we propose a new strategy to multi-label learning by leveraginglabelspeciﬁc features, where a simple yet effective algorithm named LIFT is presented. Brieﬂy, LIFT constructs features speciﬁc to each label by conducting clustering analysis on its positive and negative instances, and then performs training and testing by querying the clustering results. Extensive experiments across sixteen diversiﬁed data sets clearly validate the superiority of LIFT against other wellestablished multi-label learning algorithms.

Paper link : https://doi.org/10.1109/tpami.2014.2339815
