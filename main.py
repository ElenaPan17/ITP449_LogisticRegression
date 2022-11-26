""" Elena Pan
    ITP-449
    Assignment 6
    RMS Titanic
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


def main():
    # read the dataset into a dataframe
    file = 'titanicTrain.csv'
    df = pd.read_csv(file)

    # determine the target varaible: 'Survived' as y
    # determine variables and remove the rest
    df = df[ ['Survived', 'Pclass', 'Sex', 'Age' ]]

    # check whether there're missing values
    # print(df.isnull().sum())
    df_dropped = df.dropna(subset='Age')
    # print(df_dropped.isnull().sum())

    # plot the histograms of all of the variables in a 2x2 figure
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].hist(df_dropped['Survived'])
    ax[0, 1].hist(df_dropped['Pclass'])
    ax[1, 0].hist(df_dropped['Sex'])
    ax[1, 1].hist(df_dropped['Age'])
    ax[0, 0].set(xlabel='Survived', ylabel = 'Count')
    ax[0, 1].set(xlabel='Pclass', ylabel = 'Count')
    ax[1, 0].set(xlabel='Sex',ylabel = 'Count')
    ax[1, 1].set(xlabel='Age', ylabel = 'Count')
    
    plt.suptitle('Titanic Data: Histograms of Input Variables')
    plt.tight_layout()
    plt.savefig('Histograms.png')

    # convert all categorical variables into dummy variables
    df_titanic = pd.get_dummies(df_dropped)
    
    # train the data and then fit the trained data to a logistic regression model
    y = df_titanic['Survived']
    X = df_titanic.drop('Survived', axis = 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=76)
    model_logreg = LogisticRegression(max_iter=500)
    model_logreg.fit(X_train, y_train)

    # calculate the accuracy of the y_pred
    y_pred = model_logreg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # plot the confusion matrix 
    cm = confusion_matrix(y_test, y_pred)
    cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model_logreg.classes_)
    fig, ax1 = plt.subplots()
    cm_disp.plot(ax=ax1)
    ax1.xaxis.set_ticklabels(['No', 'Yes'])
    ax1.yaxis.set_ticklabels(['No', 'Yes'])
    plt.suptitle('Titanic Survivability' + '\n(Model Accuracy: ' + str("%.2f" % (accuracy*100)) + '%)')
    plt.savefig('confusion matrix.png')

    # display the predicted value of the survivability of a 30 year old male passenger traveling in 3rd class
    d = {'Pclass': [3], 'Age': [30.0], 'Sex_female': [0], 'Sex_male': [1]}
    df_new = pd.DataFrame(data = d)
    y_new_pred = model_logreg.predict(df_new)
    print('Prediction for 30-year-old male passenger in 3rd class:', y_new_pred[0])


if __name__ == '__main__':
    main()
