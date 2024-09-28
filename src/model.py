import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
import lightgbm as lgb


def train_model(X, y):
    """
    Trains a stacking model with multiple sub-models.

    Parameters:
    X (pd.DataFrame): Training features.
    y (pd.Series): Training labels (target).

    Returns:
    StackingClassifier: The trained stacking model.
    """

    # Definition of sub-models for stacking
    estimators = [
        (
            "rf",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_leaf=2,
                min_samples_split=2,
                random_state=42,
            ),
        ),
        (
            "logreg",
            LogisticRegression(
                C=1, penalty="l2", solver="saga", max_iter=200, random_state=42
            ),
        ),
        ("svm", SVC(kernel="rbf", gamma=0.1, C=10, random_state=42)),
        (
            "lgb",
            lgb.LGBMClassifier(
                boosting_type="gbdt",
                learning_rate=0.1,
                max_depth=5,
                n_estimators=100,
                num_leaves=31,
                random_state=42,
            ),
        ),
    ]

    # Parameters for grid search for logistic regression
    param_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear", "saga"],
        "max_iter": [100, 200, 300],
    }

    # Grid search for logistic regression as the final estimator
    logreg = LogisticRegression(random_state=42)
    logreg_grid_search = GridSearchCV(
        estimator=logreg, param_grid=param_grid, cv=5, scoring="accuracy", n_jobs=-1
    )

    # Stacking model
    stack = StackingClassifier(
        estimators=estimators, final_estimator=logreg_grid_search, cv=5, n_jobs=-1
    )

    # Training the model
    stack.fit(X, y)

    return stack


def predict_model(model, X_test):
    """
    Uses a previously trained model to make predictions on new data.

    Parameters:
    model (StackingClassifier): The trained stacking model.
    X_test (pd.DataFrame): The input data for which we want to make predictions.

    Returns:
    np.ndarray: The predictions for the X_test data.
    """

    return model.predict(X_test)


def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model using test data.

    Parameters:
    model (StackingClassifier): The trained stacking model.
    X_test (pd.DataFrame): The test data (features).
    y_test (pd.Series): The true labels for the test data.

    Returns:
    float: The accuracy score of the model on the test data.
    """

    return model.score(X_test, y_test)


def save_submission(
    predictions, passenger_ids, output_path="submission/submission.csv"
):
    """
    Saves the predictions to a CSV file for submission to Kaggle.

    Parameters:
    predictions (np.ndarray): The predictions of the survivors.
    passenger_ids (pd.Series): The passenger IDs.
    output_path (str): The path to save the submission file.
    """

    submission = pd.DataFrame({"PassengerId": passenger_ids, "Survived": predictions})

    submission.to_csv(output_path, index=False)
    print(f"Submission file saved to {output_path}")
