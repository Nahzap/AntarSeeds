import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, Dict, Any
import logging

logger = logging.getLogger(__name__)

def save_plot_with_data(output_path: Union[str, Path], data: Union[pd.DataFrame, Dict[str, Any]], **kwargs):
    """
    Saves the current matplotlib figure to output_path and also saves the provided data 
    as a CSV file with the same base name.
    
    Args:
        output_path: Path to save the .png image
        data: DataFrame or dict of arrays representing the plotted data.
        **kwargs: arguments passed to plt.savefig (e.g. dpi, bbox_inches)
    """
    output_path = Path(output_path)
    
    plt.savefig(output_path, **kwargs)
    
    csv_path = output_path.with_suffix(".csv")
    
    try:
        if isinstance(data, pd.DataFrame):
            data.to_csv(csv_path, index=False)
        elif isinstance(data, dict):
            try:
                df = pd.DataFrame(data)
            except ValueError:
                df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data.items()]))
            df.to_csv(csv_path, index=False)
        else:
            logger.warning(f"Data must be a pandas DataFrame or a dict. Got {type(data)}.")
    except Exception as e:
        logger.warning(f"Failed to save data for {output_path} to CSV: {e}")

