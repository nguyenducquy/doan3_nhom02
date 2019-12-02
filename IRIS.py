from sklearn.datasets import load_iris
iris = load_iris()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( iris.data, iris.target, test_size = 0.3)

from sklearn import tree
clf = tree.DecisionTreeClassifier(max_depth=3)
#max_depth = 3 là độ sâu của cây quyết định
clf = clf.fit(X_train, y_train)

predicted = clf.predict(X_test)


from sklearn.metrics import  confusion_matrix
cm=confusion_matrix(y_test, predicted);
acc1=sum(predicted==y_test)/float(len(y_test))

tree.export_graphviz(clf, out_file="tr.dot",
                        feature_names=iris.feature_names,
                        class_names=iris.target_names,
                        filled=True, rounded=True)


from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,predicted)
#print(acc*100)

import pydotplus
dot_data = tree.export_graphviz(clf , out_file = None , filled = True , rounded = True , special_characters = True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("graph.png")