"""Feature extraction utilities for HRV analysis."""
import numpy as np
from typing import Dict, List


class FeatureExtractor:
    """Static class for extracting HRV features from biosignal data."""
    
    @staticmethod
    def extract_hr_mean(hr_values: List[float]) -> float:
        """Extract mean heart rate from HR values.
        
        Args:
            hr_values: List of heart rate values in BPM
            
        Returns:
            Mean heart rate, or 0.0 if empty list
        """
        if not hr_values:
            return 0.0
        return float(np.mean(hr_values))
    
    @staticmethod
    def extract_sdnn(rr_intervals: List[float]) -> float:
        """Extract SDNN (standard deviation of NN intervals) from RR intervals.
        
        Args:
            rr_intervals: List of RR intervals in milliseconds
            
        Returns:
            SDNN value
        """
        cleaned = FeatureExtractor._clean_rr_intervals(rr_intervals)
        if len(cleaned) < 2:
            return 0.0
        return float(np.std(cleaned, ddof=1))
    
    @staticmethod
    def extract_rmssd(rr_intervals: List[float]) -> float:
        """Extract RMSSD (root mean square of successive differences) from RR intervals.
        
        Args:
            rr_intervals: List of RR intervals in milliseconds
            
        Returns:
            RMSSD value
        """
        cleaned = FeatureExtractor._clean_rr_intervals(rr_intervals)
        if len(cleaned) < 2:
            return 0.0
        arr = np.array(cleaned)
        diffs = np.diff(arr)
        return float(np.sqrt(np.mean(diffs ** 2)))
    
    @staticmethod
    def extract_features(hr_values: List[float], rr_intervals: List[float]) -> Dict[str, float]:
        """Extract all features from HR values and RR intervals.
        
        Args:
            hr_values: List of heart rate values in BPM
            rr_intervals: List of RR intervals in milliseconds
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # HR mean
        features["hr_mean"] = FeatureExtractor.extract_hr_mean(hr_values)
        
        # RR interval features
        features["sdnn"] = FeatureExtractor.extract_sdnn(rr_intervals)
        features["rmssd"] = FeatureExtractor.extract_rmssd(rr_intervals)
        
        return features
    
    @staticmethod
    def _clean_rr_intervals(rr_intervals: List[float]) -> List[float]:
        """Clean RR intervals by removing physiologically invalid values.
        
        Args:
            rr_intervals: List of RR intervals in milliseconds
            
        Returns:
            Cleaned list of RR intervals
        """
        MIN_VALID_RR_MS = 300.0  # 200 BPM
        MAX_VALID_RR_MS = 2000.0  # 30 BPM
        return [rr for rr in rr_intervals if MIN_VALID_RR_MS <= rr <= MAX_VALID_RR_MS]
    
    @staticmethod
    def validate_features(features: Dict[str, float], required: List[str]) -> bool:
        """Validate that all required features are present.
        
        Args:
            features: Dictionary of features
            required: List of required feature names
            
        Returns:
            True if all required features are present, False otherwise
        """
        return all(key in features for key in required)
    
    @staticmethod
    def normalize_features(
        features: Dict[str, float],
        mu: Dict[str, float],
        sigma: Dict[str, float]
    ) -> Dict[str, float]:
        """Normalize features using mean (mu) and standard deviation (sigma).
        
        Args:
            features: Dictionary of features to normalize
            mu: Dictionary of mean values for each feature
            sigma: Dictionary of standard deviation values for each feature
            
        Returns:
            Dictionary of normalized features
        """
        normalized = {}
        for key, value in features.items():
            if key in mu and key in sigma:
                if sigma[key] > 0:
                    normalized[key] = (value - mu[key]) / sigma[key]
                else:
                    normalized[key] = 0.0
            else:
                normalized[key] = value
        return normalized

