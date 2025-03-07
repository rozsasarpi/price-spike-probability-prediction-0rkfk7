"""
Defines configuration classes for managing price threshold values used in RTLMP spike prediction.

This module provides functionality to define, validate, and retrieve threshold values
that determine what constitutes a price spike in the ERCOT market.
"""

from typing import Dict, List, Optional, Union, Tuple, Set, Any
import numpy as np  # version 1.24+
import pydantic  # version 2.0+

# Internal imports
from ..utils.type_definitions import ThresholdValue, NodeID
from ..utils.logging import get_logger
from ..config.validation import validate_config

# Set up logger
logger = get_logger(__name__)

# Default threshold values for RTLMP spike prediction (in $/MWh)
DEFAULT_THRESHOLDS = [50.0, 100.0, 200.0, 500.0, 1000.0]


def validate_thresholds(thresholds: List[ThresholdValue]) -> List[ThresholdValue]:
    """
    Validates a list of threshold values to ensure they are valid for price spike prediction.
    
    Args:
        thresholds: List of threshold values to validate
        
    Returns:
        Validated list of threshold values
        
    Raises:
        ValueError: If thresholds are invalid
    """
    # Check that thresholds is a non-empty list
    if not thresholds:
        logger.warning("Empty threshold list provided, using default thresholds")
        return DEFAULT_THRESHOLDS.copy()
    
    # Validate that all threshold values are positive numbers
    for threshold in thresholds:
        if not isinstance(threshold, (int, float)):
            raise ValueError(f"Threshold must be a number: {threshold}")
        if threshold <= 0:
            raise ValueError(f"Threshold must be positive: {threshold}")
    
    # Remove any duplicate threshold values
    unique_thresholds = list(set(thresholds))
    
    # Ensure thresholds are in ascending order
    validated_thresholds = sorted(unique_thresholds)
    
    # Log a warning if thresholds were modified
    if validated_thresholds != thresholds:
        logger.warning(
            f"Thresholds were modified during validation: "
            f"Original={thresholds}, Validated={validated_thresholds}"
        )
    
    return validated_thresholds


def get_default_thresholds() -> List[ThresholdValue]:
    """
    Returns the default set of threshold values for RTLMP spike prediction.
    
    Returns:
        List of default threshold values
    """
    return DEFAULT_THRESHOLDS.copy()


class ThresholdConfig:
    """
    Configuration class for managing price threshold values.
    
    This class provides functionality to define, retrieve, and manage threshold
    values that determine what constitutes a price spike in the ERCOT market.
    Thresholds can be set globally or customized for specific nodes.
    """
    
    def __init__(self, default_thresholds: Optional[List[ThresholdValue]] = None):
        """
        Initializes ThresholdConfig with default or custom threshold values.
        
        Args:
            default_thresholds: Optional list of default threshold values.
                               If None, DEFAULT_THRESHOLDS will be used.
        """
        # If default_thresholds is None, use DEFAULT_THRESHOLDS
        if default_thresholds is None:
            default_thresholds = DEFAULT_THRESHOLDS.copy()
        
        # Validate the provided default thresholds
        self._default_thresholds = validate_thresholds(default_thresholds)
        
        # Initialize empty _node_thresholds dictionary
        self._node_thresholds: Dict[NodeID, List[ThresholdValue]] = {}
    
    def get_thresholds(self, node_id: Optional[NodeID] = None) -> List[ThresholdValue]:
        """
        Returns threshold values for a specific node or default thresholds.
        
        Args:
            node_id: Optional node identifier. If None, default thresholds are returned.
                    If provided and exists in _node_thresholds, node-specific thresholds are returned.
                    Otherwise, default thresholds are returned.
                    
        Returns:
            List of threshold values
        """
        # If node_id is None, return _default_thresholds
        if node_id is None:
            return self._default_thresholds.copy()
        
        # If node_id exists in _node_thresholds, return those thresholds
        if node_id in self._node_thresholds:
            return self._node_thresholds[node_id].copy()
        
        # Otherwise, return _default_thresholds
        return self._default_thresholds.copy()
    
    def get_default_threshold(self) -> ThresholdValue:
        """
        Returns the default threshold value (typically used for single-threshold operations).
        
        Returns:
            Default threshold value. This is the middle value from _default_thresholds,
            or the lower middle value if _default_thresholds has even length.
        """
        if not self._default_thresholds:
            # This should not happen due to validation, but just in case
            logger.warning("No default thresholds defined, using 100.0")
            return 100.0
        
        # Return the middle value from _default_thresholds
        middle_index = len(self._default_thresholds) // 2
        return self._default_thresholds[middle_index]
    
    def add_custom_threshold(self, threshold: ThresholdValue, node_id: Optional[NodeID] = None) -> bool:
        """
        Adds a custom threshold value for a specific node or to default thresholds.
        
        Args:
            threshold: Threshold value to add
            node_id: Optional node identifier. If None, threshold is added to default thresholds.
                    Otherwise, threshold is added to node-specific thresholds.
                    
        Returns:
            True if threshold was added successfully, False otherwise
        """
        try:
            # Validate that threshold is a positive number
            if not isinstance(threshold, (int, float)) or threshold <= 0:
                logger.error(f"Invalid threshold value: {threshold}")
                return False
            
            if node_id is None:
                # Add threshold to _default_thresholds if not already present
                if threshold not in self._default_thresholds:
                    self._default_thresholds.append(threshold)
                    # Re-sort the threshold list to maintain ascending order
                    self._default_thresholds.sort()
                    logger.debug(f"Added threshold {threshold} to default thresholds")
                return True
            else:
                # Initialize node thresholds if needed
                if node_id not in self._node_thresholds:
                    self._node_thresholds[node_id] = self._default_thresholds.copy()
                
                # Add threshold to node thresholds if not already present
                if threshold not in self._node_thresholds[node_id]:
                    self._node_thresholds[node_id].append(threshold)
                    # Re-sort the threshold list to maintain ascending order
                    self._node_thresholds[node_id].sort()
                    logger.debug(f"Added threshold {threshold} to node {node_id}")
                return True
        except Exception as e:
            logger.error(f"Error adding threshold: {e}")
            return False
    
    def remove_threshold(self, threshold: ThresholdValue, node_id: Optional[NodeID] = None) -> bool:
        """
        Removes a threshold value from a specific node or default thresholds.
        
        Args:
            threshold: Threshold value to remove
            node_id: Optional node identifier. If None, threshold is removed from default thresholds.
                    Otherwise, threshold is removed from node-specific thresholds.
                    
        Returns:
            True if threshold was removed successfully, False otherwise
        """
        try:
            if node_id is None:
                # Remove threshold from _default_thresholds if present
                if threshold in self._default_thresholds:
                    self._default_thresholds.remove(threshold)
                    logger.debug(f"Removed threshold {threshold} from default thresholds")
                    return True
                return False
            else:
                # Check if node exists in _node_thresholds
                if node_id not in self._node_thresholds:
                    logger.warning(f"Node {node_id} does not have custom thresholds")
                    return False
                
                # Remove threshold from node thresholds if present
                if threshold in self._node_thresholds[node_id]:
                    self._node_thresholds[node_id].remove(threshold)
                    logger.debug(f"Removed threshold {threshold} from node {node_id}")
                    
                    # If node thresholds is now empty, remove the node entry
                    if not self._node_thresholds[node_id]:
                        del self._node_thresholds[node_id]
                        logger.debug(f"Removed empty threshold list for node {node_id}")
                    
                    return True
                return False
        except Exception as e:
            logger.error(f"Error removing threshold: {e}")
            return False
    
    def set_thresholds(self, thresholds: List[ThresholdValue], node_id: Optional[NodeID] = None) -> bool:
        """
        Sets the complete list of thresholds for a specific node or default thresholds.
        
        Args:
            thresholds: List of threshold values to set
            node_id: Optional node identifier. If None, _default_thresholds is set.
                    Otherwise, node-specific thresholds are set.
                    
        Returns:
            True if thresholds were set successfully, False otherwise
        """
        try:
            # Validate the provided thresholds
            validated_thresholds = validate_thresholds(thresholds)
            
            if node_id is None:
                # Set _default_thresholds to validated thresholds
                self._default_thresholds = validated_thresholds
                logger.debug(f"Set default thresholds to {validated_thresholds}")
                return True
            else:
                # Set node-specific thresholds
                self._node_thresholds[node_id] = validated_thresholds
                logger.debug(f"Set thresholds for node {node_id} to {validated_thresholds}")
                return True
        except Exception as e:
            logger.error(f"Error setting thresholds: {e}")
            return False
    
    def get_all_nodes(self) -> List[NodeID]:
        """
        Returns a list of all nodes with custom thresholds.
        
        Returns:
            List of node identifiers
        """
        return list(self._node_thresholds.keys())
    
    def get_all_thresholds(self) -> Set[ThresholdValue]:
        """
        Returns a set of all unique threshold values across all nodes.
        
        Returns:
            Set of unique threshold values
        """
        # Initialize an empty set
        all_thresholds = set(self._default_thresholds)
        
        # Add all threshold values from all nodes in _node_thresholds
        for node_thresholds in self._node_thresholds.values():
            all_thresholds.update(node_thresholds)
        
        return all_thresholds
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the threshold configuration to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the threshold configuration
        """
        # Create a dictionary with 'default_thresholds' key
        config_dict = {
            'default_thresholds': self._default_thresholds,
        }
        
        # Add 'node_thresholds' key if there are any node-specific thresholds
        if self._node_thresholds:
            config_dict['node_thresholds'] = {
                node_id: thresholds 
                for node_id, thresholds in self._node_thresholds.items()
            }
        
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ThresholdConfig':
        """
        Creates a ThresholdConfig instance from a dictionary representation.
        
        Args:
            config_dict: Dictionary representation of threshold configuration
            
        Returns:
            ThresholdConfig instance
        """
        # Extract 'default_thresholds' from config_dict
        default_thresholds = config_dict.get('default_thresholds', DEFAULT_THRESHOLDS)
        
        # Create a new ThresholdConfig instance
        config = cls(default_thresholds=default_thresholds)
        
        # Add node-specific thresholds if present
        if 'node_thresholds' in config_dict:
            for node_id, thresholds in config_dict['node_thresholds'].items():
                config.set_thresholds(thresholds, node_id)
        
        return config


class DynamicThresholdConfig(ThresholdConfig):
    """
    Extended threshold configuration with dynamic threshold calculation capabilities.
    
    This class extends ThresholdConfig to support dynamic threshold calculation
    based on input data, allowing for adaptive threshold values that can change
    based on market conditions or other factors.
    """
    
    def __init__(self, default_thresholds: Optional[List[ThresholdValue]] = None):
        """
        Initializes DynamicThresholdConfig with default thresholds and optional calculators.
        
        Args:
            default_thresholds: Optional list of default threshold values.
                              If None, DEFAULT_THRESHOLDS will be used.
        """
        # Call parent class constructor
        super().__init__(default_thresholds=default_thresholds)
        
        # Initialize empty dictionary for threshold calculators
        self._threshold_calculators: Dict[str, Callable[[Any], List[ThresholdValue]]] = {}
    
    def add_dynamic_threshold(self, calculator_name: str, calculator_func: Callable[[Any], List[ThresholdValue]]) -> bool:
        """
        Adds a dynamic threshold calculator function.
        
        Args:
            calculator_name: Name to identify the calculator
            calculator_func: Function that takes input data and returns a list of threshold values
            
        Returns:
            True if calculator was added successfully, False otherwise
        """
        try:
            # Validate that calculator_func is callable
            if not callable(calculator_func):
                logger.error(f"Calculator function must be callable: {calculator_func}")
                return False
            
            # Add calculator_func to _threshold_calculators
            self._threshold_calculators[calculator_name] = calculator_func
            logger.debug(f"Added threshold calculator: {calculator_name}")
            return True
        except Exception as e:
            logger.error(f"Error adding threshold calculator: {e}")
            return False
    
    def remove_dynamic_threshold(self, calculator_name: str) -> bool:
        """
        Removes a dynamic threshold calculator.
        
        Args:
            calculator_name: Name of the calculator to remove
            
        Returns:
            True if calculator was removed successfully, False otherwise
        """
        try:
            # Remove calculator from _threshold_calculators if it exists
            if calculator_name in self._threshold_calculators:
                del self._threshold_calculators[calculator_name]
                logger.debug(f"Removed threshold calculator: {calculator_name}")
                return True
            
            logger.warning(f"Threshold calculator not found: {calculator_name}")
            return False
        except Exception as e:
            logger.error(f"Error removing threshold calculator: {e}")
            return False
    
    def calculate_thresholds(self, calculator_name: str, input_data: Any) -> List[ThresholdValue]:
        """
        Calculates dynamic thresholds using a named calculator and input data.
        
        Args:
            calculator_name: Name of the calculator to use
            input_data: Input data for threshold calculation
            
        Returns:
            Dynamically calculated threshold values
            
        Raises:
            ValueError: If calculator_name doesn't exist
        """
        # Check if calculator_name exists in _threshold_calculators
        if calculator_name not in self._threshold_calculators:
            logger.error(f"Threshold calculator not found: {calculator_name}")
            logger.info(f"Returning default thresholds")
            return self._default_thresholds.copy()
        
        try:
            # Call the calculator function with input_data
            calculator = self._threshold_calculators[calculator_name]
            thresholds = calculator(input_data)
            
            # Validate the returned thresholds
            validated_thresholds = validate_thresholds(thresholds)
            
            return validated_thresholds
        except Exception as e:
            logger.error(f"Error calculating thresholds with {calculator_name}: {e}")
            return self._default_thresholds.copy()
    
    def get_available_calculators(self) -> List[str]:
        """
        Returns a list of available dynamic threshold calculator names.
        
        Returns:
            List of calculator names
        """
        return list(self._threshold_calculators.keys())
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the dynamic threshold configuration to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the configuration
        """
        # Call parent class to_dict method
        config_dict = super().to_dict()
        
        # Add 'dynamic_calculators' key with list of calculator names
        if self._threshold_calculators:
            config_dict['dynamic_calculators'] = list(self._threshold_calculators.keys())
        
        return config_dict