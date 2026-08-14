"""fraud-monitoring — drift detection, performance monitoring, and retraining
policy for the credit-card fraud model produced by Credit-Card-Fraud-testing.

Deliberately a separate repository from the training pipeline: the two have
different cadences (training runs on demand, monitoring runs on a schedule),
different failure modes, and different reviewers. The only coupling is the
artifact contract — model.pkl / scaler.pkl / threshold.json.
"""

__version__ = "0.1.0"
