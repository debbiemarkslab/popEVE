import numpy as np
import pandas as pd
import torch

def get_training_and_holdout_data_from_processed_file(training_data_df, transform_data=True, training_frac=1, random_state=42, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Extracts training and holdout data from a processed file.

    Parameters:
    - training_data_df: DataFrame, processed data containing 'model_score', 'observed', and 'mutant' columns.
    - transform_data: bool, flag to indicate whether to transform the data to a specific range.
    - training_frac: float, fraction of data to use for training.
    - random_state: int, seed for reproducibility.

    Returns:
    Tuple of PyTorch tensors and additional information:
    (train_x, train_y, variants_train, test_x, test_y, variants_test, X_min, X_max)
    """
    # Split data into training and holdout sets
    train = training_data_df.sample(frac=training_frac, random_state=random_state)
    test = training_data_df.drop(train.index)

    # Extract features (X) and labels (Y) for training set
    X = train.model_score.values[np.newaxis].T.tolist()
    Y = list(train.observed)
    variants_train = train.mutant.values

    # Extract features (X) and labels (Y) for test set
    X_test = test.model_score.values[np.newaxis].T.tolist()
    Y_test = list(test.observed)
    variants_test = test.mutant.values

    # Get the min and max values of the model_score column for rescaling.
    X_max = train.model_score.max()
    X_min = train.model_score.min()

    # Optionally, transform the data to a specific range
    if transform_data:
        X_rescale = (X - X_min) / (X_max - X_min)
        X_test_rescale = (X_test - X_min) / (X_max - X_min)
    else:
        X_rescale = X
        X_test_rescale = X_test

    # Convert data to PyTorch tensors and move to the specified device
    train_x = torch.tensor(X_rescale).float().to(device)
    train_y = torch.tensor(Y).float().to(device)
    test_x = torch.tensor(X_test_rescale).float().to(device)
    test_y = torch.tensor(Y_test).float().to(device)

    # Return the data along with variant information and min/max values
    return (train_x, train_y, variants_train, test_x, test_y, variants_test, X_min, X_max)

def percentiles_from_samples(samples, percentiles=[0.05, 0.5, 0.95]):
    """
    Computes percentiles from a tensor of samples.

    Parameters:
    - samples: PyTorch tensor, samples from a distribution.
    - percentiles: list of floats, percentiles to compute.

    Returns:
    List of values corresponding to the specified percentiles.
    """
    num_samples = samples.size(0)
    samples = samples.sort(dim=0)[0]

    # Get samples corresponding to percentile
    percentile_samples = [samples[int(num_samples * percentile)] for percentile in percentiles]

    return percentile_samples

def get_scores(model, x_tensor, variants, sample_size=10**3):
    """
    Computes scores from a trained model.

    Parameters:
    - model: PyTorch model, a trained model.
    - x_tensor: PyTorch tensor, input data.
    - variants: numpy array, variant information.
    - sample_size: int, number of samples to generate from the model.

    Returns:
    DataFrame with mutant information and model scores.
    """
    model.eval()
    with torch.no_grad():
        output = model(x_tensor)

    # Generate samples from the model
    samples = output.sample(torch.Size([sample_size]))
    lower, mean, upper = percentiles_from_samples(samples)

    # Create a DataFrame with the results
    df = pd.DataFrame({'mutant': variants, 'X': torch.flatten(x_tensor).cpu(),
                       'GP_mean': mean.cpu(), 'GP_lower': lower.cpu(), 'GP_upper': upper.cpu(), 'GP_mean_all_samples': samples.mean(0).cpu(),
                       'mean_prob': mean.sigmoid().cpu()})

    return df
