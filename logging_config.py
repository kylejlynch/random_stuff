"""
Centralized Logging Configuration for Fraud Clustering Pipeline
This module provides consistent logging setup across all pipeline components.
"""

import logging
import os
from datetime import datetime
import sys

class FraudClusteringLogger:
    """Centralized logger for fraud clustering pipeline"""
    
    def __init__(self, log_dir="logs", log_level=logging.INFO):
        """
        Initialize the logging configuration
        
        Args:
            log_dir (str): Directory to store log files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.log_dir = log_dir
        self.log_level = log_level
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration with both file and console handlers"""
        
        # Create logs directory if it doesn't exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        # Create log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(self.log_dir, f"fraud_clustering_{timestamp}.log")
        
        # Clear any existing handlers to avoid duplication
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # Configure root logger
        logging.basicConfig(
            level=self.log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                # File handler - logs everything to file
                logging.FileHandler(log_filename, mode='w', encoding='utf-8'),
                # Console handler - logs to console with cleaner format
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Set console handler format to be cleaner
        console_handler = logging.root.handlers[1]  # StreamHandler is second
        console_formatter = logging.Formatter('%(levelname)s - %(name)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        
        # Log the setup
        logger = logging.getLogger(__name__)
        logger.info(f"Logging initialized. Log file: {log_filename}")
        logger.info(f"Log level: {logging.getLevelName(self.log_level)}")

def get_logger(name):
    """
    Get a logger with the specified name
    
    Args:
        name (str): Logger name (typically __name__ of the calling module)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name)

def setup_pipeline_logging(log_dir="logs", log_level=logging.INFO, verbose=False):
    """
    Setup logging for the entire fraud clustering pipeline
    
    Args:
        log_dir (str): Directory to store log files
        log_level: Logging level
        verbose (bool): If True, set console output to DEBUG level
        
    Returns:
        FraudClusteringLogger: Logger instance
    """
    if verbose:
        console_level = logging.DEBUG
    else:
        console_level = logging.INFO
    
    # Initialize the logger
    fraud_logger = FraudClusteringLogger(log_dir=log_dir, log_level=log_level)
    
    # Adjust console handler level if verbose
    if verbose and len(logging.root.handlers) > 1:
        console_handler = logging.root.handlers[1]
        console_handler.setLevel(console_level)
    
    return fraud_logger

def log_section_header(logger, title, char="="):
    """
    Log a formatted section header
    
    Args:
        logger: Logger instance
        title (str): Section title
        char (str): Character to use for decoration
    """
    separator = char * 60
    logger.info("")
    logger.info(separator)
    logger.info(title.center(60))
    logger.info(separator)

def log_step_header(logger, step_num, title, char="="):
    """
    Log a formatted step header
    
    Args:
        logger: Logger instance
        step_num (int): Step number
        title (str): Step title
        char (str): Character to use for decoration
    """
    separator = char * 50
    step_title = f"STEP {step_num}: {title}"
    logger.info("")
    logger.info(separator)
    logger.info(step_title)
    logger.info(separator)

def log_completion_summary(logger, start_time, end_time, output_files):
    """
    Log pipeline completion summary
    
    Args:
        logger: Logger instance
        start_time: Pipeline start time
        end_time: Pipeline end time
        output_files (list): List of output file paths
    """
    duration = end_time - start_time
    
    log_section_header(logger, "PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info(f"Total execution time: {duration:.2f} seconds")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    logger.info("Output Files Generated:")
    for i, file_path in enumerate(output_files, 1):
        logger.info(f"  {i}. {file_path}")

def log_error_summary(logger, error, step_name):
    """
    Log error summary with details
    
    Args:
        logger: Logger instance
        error: Exception that occurred
        step_name (str): Name of the step where error occurred
    """
    logger.error(f"Pipeline failed during {step_name}")
    logger.error(f"Error type: {type(error).__name__}")
    logger.error(f"Error message: {str(error)}")
    logger.debug(f"Full traceback:", exc_info=True)

# Example usage and configuration presets
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

def get_log_level(level_string):
    """Convert string log level to logging constant"""
    return LOG_LEVELS.get(level_string.upper(), logging.INFO)