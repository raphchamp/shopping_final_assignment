import csv
import sys
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    with open("shopping.csv", "r") as inside_file:
        dictReader = csv.DictReader(inside_file)
        evidence = []
        labels = []
        for row in dictReader:
            currEvidence = []
            currEvidence.append(int(row["Administrative"]))
            currEvidence.append(float(row["Administrative_Duration"]))
            currEvidence.append(int(row["Informational"]))
            currEvidence.append(float(row["Informational_Duration"]))
            currEvidence.append(int(row["ProductRelated"]))
            currEvidence.append(float(row["ProductRelated_Duration"]))
            currEvidence.append(float(row["BounceRates"]))
            currEvidence.append(float(row["ExitRates"]))
            currEvidence.append(float(row["PageValues"]))
            currEvidence.append(float(row["SpecialDay"]))
            currEvidence.append(datetime.strptime(row["Month"][:3], "%b").month - 1)
            currEvidence.append(int(row["OperatingSystems"]))
            currEvidence.append(int(row["Browser"]))
            currEvidence.append(int(row["Region"]))
            currEvidence.append(int(row["TrafficType"]))
            currEvidence.append(1 if row["VisitorType"] == "Returning_Visitor" else 0)
            currEvidence.append(1 if row["Weekend"] == "TRUE" else 0)
            evidence.append(currEvidence)
            labels.append(1 if row["Revenue"] == "TRUE" else 0)
    return (evidence, labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence,labels)
    return model

def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    true_positives=0
    true_negatives=0
    for label,prediction in zip(labels,predictions):
        if label==1 and prediction==1:
            true_positives+=1
        elif label==0 and prediction==0:
            true_negatives+=1
    sensitivity=true_positives/len(labels)
    specificity=true_negatives/len(labels)
    return(sensitivity,specificity)


if __name__ == "__main__":
    main()
