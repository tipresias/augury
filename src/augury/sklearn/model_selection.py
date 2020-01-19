"""Functions and classes for cross-validation and parameter tuning."""


def year_cv_split(X, year_range):
    """Split data by year for cross-validation for time-series data.

    Makes data from each year in the year_range a test set per split, with data
    from all earlier years being in the train split.
    """
    return [
        ((X["year"] < year).to_numpy(), (X["year"] == year).to_numpy())
        for year in range(*year_range)
    ]
