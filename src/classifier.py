import wittgenstein as lw
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from colorama import Fore, Style, init

# Initialize colorama for cross-platform compatibility
init(autoreset=True)


def train_RIPPER_classifier(X_train, y_train):
    """
    Train a RIPPER classifier and display the learned rules with metrics.

    Parameters:
    X_train: Features training data (numpy array or pandas DataFrame)
    y_train: Target training data (numpy array or pandas Series)

    Returns:
    tuple: (trained classifier, ruleset)
    """
    # Convert numpy arrays to DataFrame if necessary
    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train)

    # Ensure column names exist
    if X_train.columns.dtype == "int64":
        X_train.columns = [f"feature_{i}" for i in range(len(X_train.columns))]

    # Create training DataFrame
    df_train = X_train.copy()
    df_train["target"] = y_train

    # Initialize and train the RIPPER classifier
    classifier = lw.RIPPER(k=2, prune_size=0.33)  # k=2 for two optimization phases
    try:
        classifier.fit(df_train, class_feat="target", pos_class=1)
    except Exception as e:
        print(f"Error during training: {str(e)}")
        print("Trying alternative configuration...")
        # Try alternative configuration if initial training fails
        classifier = lw.RIPPER(k=1, prune_size=0.5)
        classifier.fit(df_train, class_feat="target")

    # Get the ruleset
    ruleset = classifier.ruleset_

    # Display rules and metrics
    print(Fore.CYAN + "\nDecision Rules:\n" + "=" * 50)
    for i, rule in enumerate(ruleset.rules, start=1):
        # Compute metrics for this rule
        try:
            rule_support = ruleset.support(rule, df_train, "target")
            rule_accuracy = ruleset.accuracy(rule, df_train, "target")
            rule_coverage = rule_support / len(df_train)
            false_positive_rate = 1 - rule_accuracy if rule_coverage > 0 else 0

            # Rule Header
            print(Fore.YELLOW + f"\nRule {i}:")

            # Conditions
            for condition in rule.conds:
                print(Fore.GREEN + f"  IF {condition}")

            # Class prediction
            print(Fore.MAGENTA + f"  THEN Class = {rule.class_name}")

            # Metrics
            print(
                Fore.BLUE + f"  [Support: {rule_support}, "
                f"Accuracy: {rule_accuracy:.2f}, "
                f"Coverage: {rule_coverage:.2f}, "
                f"False Positive Rate: {false_positive_rate:.2f}]"
            )
        except Exception as e:
            print(f"Error computing metrics for rule {i}: {str(e)}")

    print(Fore.CYAN + "\n" + "=" * 50)
    return classifier, ruleset


def evaluate_classifier(classifier, X_test, y_test, rules=None):
    """
    Evaluate the trained RIPPER classifier on test data.

    Parameters:
    classifier: Trained RIPPER classifier
    X_test: Features test data
    y_test: Target test data
    rules: Ruleset (optional)

    Returns:
    dict: Classification report as dictionary
    """
    # Convert numpy arrays to DataFrame if necessary
    if isinstance(X_test, np.ndarray):
        X_test = pd.DataFrame(X_test)

    # Ensure column names match training data
    if X_test.columns.dtype == "int64":
        X_test.columns = [f"feature_{i}" for i in range(len(X_test.columns))]

    # Prepare test data
    df_test = X_test.copy()

    try:
        # Make predictions
        y_pred = classifier.predict(df_test)

        # Generate and display classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Additional rule-specific evaluation if rules are provided
        if rules is not None:
            print("\nRule-specific Performance:")
            for i, rule in enumerate(rules.rules, start=1):
                matches = rules.covers(rule, df_test)
                if matches.any():
                    correct_predictions = sum(y_test[matches] == rule.class_name)
                    rule_accuracy = correct_predictions / sum(matches)
                    print(f"Rule {i} accuracy on test set: {rule_accuracy:.2f}")

        return report

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        return None
