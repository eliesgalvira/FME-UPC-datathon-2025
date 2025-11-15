"""Simple logging utility."""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
	"""Get or create a logger with the given name."""
	logger = logging.getLogger(name)
	
	if not logger.handlers:
		# Configure logger if not already configured
		handler = logging.StreamHandler()
		formatter = logging.Formatter(
			'%(asctime)s - %(name)s - %(levelname)s - %(message)s',
			datefmt='%Y-%m-%d %H:%M:%S'
		)
		handler.setFormatter(formatter)
		logger.addHandler(handler)
		logger.setLevel(logging.INFO)
	
	return logger

