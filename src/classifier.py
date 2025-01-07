import wittgenstein as lw
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from colorama import Fore, Style, init

init(autoreset=True)


def train_RIPPER_classifier(X_train, y_train, target_name="target"):
    """
    Train a RIPPER classifier and display the learned rules with metrics.

    Parameters:
    X_train: Features training data (numpy array or pandas DataFrame)
    y_train: Target training data (numpy array or pandas Series)

    Returns:
    tuple: (trained classifier, ruleset)
    """
    classifier = lw.RIPPER(k=2, prune_size=0.33, verbosity=1)
    try:
        classifier.fit(trainset=X_train, y=y_train)
    except Exception as e:
        print(f"Error during training: {str(e)}")

    ruleset = classifier.ruleset_

    df_train = X_train.copy()
    df_train[target_name] = y_train

    print(Fore.CYAN + "\nDecision Rules:\n" + "=" * 50)
    for i, rule in enumerate(ruleset.rules, start=1):
        try:
            rule_support = ruleset.support(rule, df_train, target_name)
            rule_accuracy = ruleset.accuracy(rule, df_train, target_name)
            rule_coverage = rule_support / len(df_train)
            false_positive_rate = 1 - rule_accuracy if rule_coverage > 0 else 0

            print(Fore.YELLOW + f"\nRule {i}:")

            for condition in rule.conds:
                print(Fore.GREEN + f"  IF {condition}")

            print(Fore.MAGENTA + f"  THEN Class = {rule.class_name}")

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


def cross_validate_RIPPER(X, y, n_splits=5, target_name="target"):
    """
    Perform cross-validation to extract and analyse consistent rulesets.

    Parameters:
    X: Features data (numpy array or pandas DataFrame)
    y: Target data (numpy array or pandas Series)
    n_splits: Number of folds for cross-validation
    target_name: Name of the target column

    Returns:
    list: Aggregated rulesets across all folds
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_rulesets = []

    for fold, (train_index, test_index) in enumerate(kf.split(X), start=1):
        print(Fore.CYAN + f"\nFold {fold}/{n_splits}:" + "=" * 40)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        classifier, ruleset = train_RIPPER_classifier(X_train, y_train, target_name)
        all_rulesets.append(ruleset)

    print(Fore.GREEN + "\nCross-Validation Complete\n" + "=" * 50)
    return all_rulesets


def analyse_rulesets(all_rulesets):
    """
    Analyse rulesets to identify consistent rules across folds.

    Parameters:
    all_rulesets: List of rulesets from cross-validation

    Returns:
    dict: Aggregated rule analysis
    """
    rule_summary = {}

    for fold, ruleset in enumerate(all_rulesets, start=1):
        for rule in ruleset.rules:
            rule_text = str(rule)
            if rule_text not in rule_summary:
                rule_summary[rule_text] = {"support": [], "accuracy": [], "folds": []}

            rule_summary[rule_text]["folds"].append(fold)

    print(Fore.YELLOW + "\nAggregated Rule Analysis:\n" + "=" * 50)
    for rule_text, metrics in rule_summary.items():
        print(Fore.CYAN + f"\nRule: {rule_text}")
        print(Fore.GREEN + f"  Appeared in folds: {metrics['folds']}")

    return rule_summary
