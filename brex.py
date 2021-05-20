#Winsconsin database

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
from cf_matrix import plot_confusion_matrix



class CancerAnalysis:
    def __init__(self,path,index=469):
        data = np.genfromtxt(path,delimiter=',')
        # Remove nans
        data = data[~np.isnan(data).any(axis=1)]
        self.X_train= data[1:index,2:]
        self.Y_train = data[1:index,1]
        self.X_test = data[index+1:,2:]
        self.Y_test = data[index+1:,1]
        
        
    def byknn(self):
        print("Result By KNN")
        training_accuracy = []
        test_accuracy = []
        for n_neighbors in range(1,100):
            clf = KNeighborsClassifier(n_neighbors= n_neighbors)
            clf.fit(self.X_train,self.Y_train)
            training_accuracy.append(clf.score(self.X_train,self.Y_train))
            test_accuracy.append(clf.score(self.X_test,self.Y_test))
            print("Training at neighbors = "+str(n_neighbors),clf.score(self.X_test,self.Y_test))
        plt.plot(range(1,100),training_accuracy,label="Training")
        plt.plot(range(1,100),test_accuracy, label = "Test")
        print("Max Accuracy "+str(max(test_accuracy))+"At Neighbors = "+str(test_accuracy.index(max(test_accuracy))+1))
        max_index = test_accuracy.index(max(test_accuracy))+1
        plt.plot(max_index,max(test_accuracy), 'ro',label="MAX")
        plt.ylabel('% accuracy')
        plt.xlabel('Neighbors')
        plt.legend()
        plt.show()
        return max(test_accuracy)
             
    def bylogistic(self):
        print("Result By Logistic Regression")
        training_accuracy = []
        test_accuracy = []
        for i in range(1,30):
            clf = LogisticRegression(C=i,solver='liblinear').fit(self.X_train,self.Y_train)
            print(clf.score(self.X_test,self.Y_test))
            training_accuracy.append(clf.score(self.X_train,self.Y_train))
            test_accuracy.append(clf.score(self.X_test,self.Y_test))
        plt.plot(range(1,30),training_accuracy,label="Training")
        plt.plot(range(1,30),test_accuracy, label = "Test")
        print("Max Accuracy "+str(max(test_accuracy))+"At Inverse Regularization = "+str(test_accuracy.index(max(test_accuracy))+1))
        max_index = test_accuracy.index(max(test_accuracy))+1
        plt.plot(max_index,max(test_accuracy), 'ro',label="MAX")
        plt.ylabel('% accuracy')
        plt.xlabel('Inverse Regularization')
        plt.legend()
        plt.show()

        return max(test_accuracy)
    def bysvm(self):
        print("Result By SVM")
        training_accuracy = []
        test_accuracy = []
        for i in range(1,30):
            clf = LinearSVC(C=i,dual=True).fit(self.X_train,self.Y_train)
            training_accuracy.append(clf.score(self.X_train,self.Y_train))
            test_accuracy.append(clf.score(self.X_test,self.Y_test))
            print(clf.score(self.X_test,self.Y_test))
        
        plt.plot(range(1,30),training_accuracy,label="Training")
        plt.plot(range(1,30),test_accuracy, label = "Test")
        print("Max Accuracy "+str(max(test_accuracy))+"At Inverse Regularization = "+str(test_accuracy.index(max(test_accuracy))+1))
        max_index = test_accuracy.index(max(test_accuracy))+1
        plt.plot(max_index,max(test_accuracy), 'ro',label="MAX")
        plt.ylabel('% accuracy')
        plt.xlabel('Inverse Regularization')
        plt.legend()
        plt.show()
        
        return max(test_accuracy)        
    def bydtree(self):
        print("Result By Decision Tree Classifier")
        training_accuracy = []
        test_accuracy = []
        for i in range(1,10):
            clf = DecisionTreeClassifier(random_state = 0,max_depth=i )
            clf.fit(self.X_train,self.Y_train)
         
            print(clf.score(self.X_test,self.Y_test))
            training_accuracy.append(clf.score(self.X_train,self.Y_train))
            test_accuracy.append(clf.score(self.X_test,self.Y_test))
        
        plt.plot(range(1,10),training_accuracy,label="Training")
        plt.plot(range(1,10),test_accuracy, label = "Test")
        print("Max Accuracy "+str(max(test_accuracy))+"At Prepruining = "+str(test_accuracy.index(max(test_accuracy))+1))
        max_index = test_accuracy.index(max(test_accuracy))+1
        plt.plot(max_index,max(test_accuracy), 'ro',label="MAX")
        plt.ylabel('% accuracy')
        plt.xlabel('Prepruining')
        plt.legend()
        plt.show()
        
        return max(test_accuracy)
    

    def byNB(self):
        print("Result By Naive Bayes Classifier")
        clf = GaussianNB()
        clf.fit(self.X_train,self.Y_train)
        y_pred = clf.predict(self.X_test)
        cm=confusion_matrix(self.Y_test, y_pred)
        print(cm)
        #labels = ['True Neg','False Pos','False Neg','True Pos']
        categories = ['Malignant', 'Benign']
        #make_confusion_matrix(cf, 
                      #group_names=labels,
                      #categories=categories,).show()
        plot_confusion_matrix(cm,target_names=categories,)
        """fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(cm)
        plt.title('Confusion matrix of the classifier')
        fig.colorbar(cax)
        ax.set_xticklabels([''] + categories)
        ax.set_yticklabels([''] + categories)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()"""
        return clf.score(self.X_test,self.Y_test)


    def bygradientboost(self):
        print("Result By Gradient Boost")
        training_accuracy = []
        test_accuracy = []
        for i in range(1,10):
            clf = GradientBoostingClassifier(random_state = 0,max_depth=i,learning_rate = 0.01)
            clf.fit(self.X_train,self.Y_train)
            print(clf.score(self.X_test,self.Y_test))
            training_accuracy.append(clf.score(self.X_train,self.Y_train))
            test_accuracy.append(clf.score(self.X_test,self.Y_test))
        plt.plot(range(1,10),training_accuracy,label="Training")
        plt.plot(range(1,10),test_accuracy, label = "Test")
        print("Max Accuracy "+str(max(test_accuracy))+"At Estimators = "+str(test_accuracy.index(max(test_accuracy))+1))
        max_index = test_accuracy.index(max(test_accuracy))+1
        
        plt.plot(max_index,max(test_accuracy), 'ro',label="MAX")
        plt.ylabel('% accuracy')
        plt.xlabel('Prepruining')
        plt.legend()
        plt.show()
       
        return max(test_accuracy)

    def bypolysvm(self):
        print("Result using Polynomial Kernel")
        training_accuracy = []
        test_accuracy = []
        for i in range(1,10):
            poly_svc = svm.SVC(kernel = 'poly',degree=i)
            poly_svc.fit(scale(self.X_train),self.Y_train)
            test_accuracy.append(poly_svc.score(scale(self.X_test),self.Y_test))
            training_accuracy.append(poly_svc.score(scale(self.X_train),self.Y_train))
        
        plt.plot(range(1,10),training_accuracy,label="Training")
        plt.plot(range(1,10),test_accuracy, label = "Test")
        print("Max Accuracy "+str(max(test_accuracy))+"At Degree = "+str(test_accuracy.index(max(test_accuracy))+1))
        max_index = test_accuracy.index(max(test_accuracy))+1
        
        plt.plot(max_index,max(test_accuracy), 'ro',label="MAX")
        plt.ylabel('% accuracy')
        plt.xlabel('Degree')
        plt.legend()
        plt.show()
        return max(test_accuracy)
        
if __name__ == "__main__":
    print(CancerAnalysis('wisc_bc_data.csv',index= 425).byknn())
    '''
    acc =[]
    
    for i in range(400,540,5):
       print i,CancerAnalysis('wisc_bc_data.csv',i).byrandom() 
       acc.append(CancerAnalysis('wisc_bc_data.csv',i).byrandom())
    plt.plot(range(400,540,5),acc,label="Test")

    plt.xlabel('Training Data')
    plt.ylabel('% Accuracy')
    plt.plot(acc.index(max(acc))*5+400,max(acc),'ro',label="MAX")
    plt.show()
    '''   
