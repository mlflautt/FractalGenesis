#!/usr/bin/env python3
"""
Fractal Preference Learning System

This module provides the foundation for training neural networks to learn
user preferences from fractal selection data, eventually allowing the system
to predict which fractals a user will prefer.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import time
from datetime import datetime
import shutil


class FractalFeatureExtractor:
    """
    Extract numerical features from fractal parameters for ML training.
    
    Converts various fractal parameter formats into standardized
    numerical feature vectors that can be used by machine learning models.
    """
    
    def __init__(self):
        self.feature_names = []
        self.scalers = {}
        self.encoders = {}
        self.logger = logging.getLogger(__name__)
    
    def extract_flam3_features(self, fractal_data: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from Flam3 fractal parameters.
        
        Args:
            fractal_data: Dictionary containing Flam3 fractal parameters
            
        Returns:
            Feature vector as numpy array
        """
        features = []
        feature_names = []
        
        # Basic flame parameters
        if 'scale' in fractal_data:
            features.append(float(fractal_data['scale']))
            feature_names.append('scale')
        
        if 'rotate' in fractal_data:
            features.append(float(fractal_data['rotate']))
            feature_names.append('rotate')
        
        if 'center' in fractal_data:
            center_parts = str(fractal_data['center']).split()
            if len(center_parts) >= 2:
                features.extend([float(center_parts[0]), float(center_parts[1])])
                feature_names.extend(['center_x', 'center_y'])
            else:
                features.extend([0.0, 0.0])
                feature_names.extend(['center_x', 'center_y'])
        
        if 'brightness' in fractal_data:
            features.append(float(fractal_data['brightness']))
            feature_names.append('brightness')
        
        if 'gamma' in fractal_data:
            features.append(float(fractal_data['gamma']))
            feature_names.append('gamma')
        
        if 'vibrancy' in fractal_data:
            features.append(float(fractal_data['vibrancy']))
            feature_names.append('vibrancy')
        
        # Xform features - aggregate statistics
        if 'xforms' in fractal_data and fractal_data['xforms']:
            xforms = fractal_data['xforms']
            
            # Number of xforms
            features.append(len(xforms))
            feature_names.append('num_xforms')
            
            # Weight statistics
            weights = [xform.get('weight', 0) for xform in xforms]
            if weights:
                features.extend([
                    np.mean(weights),
                    np.std(weights),
                    np.min(weights),
                    np.max(weights)
                ])
                feature_names.extend(['weight_mean', 'weight_std', 'weight_min', 'weight_max'])
            
            # Color statistics
            colors = [xform.get('color', 0) for xform in xforms]
            if colors:
                features.extend([
                    np.mean(colors),
                    np.std(colors)
                ])
                feature_names.extend(['color_mean', 'color_std'])
            
            # Variation diversity - count unique variations used
            all_variations = set()
            for xform in xforms:
                if 'variations' in xform:
                    all_variations.update(xform['variations'].keys())
            
            features.append(len(all_variations))
            feature_names.append('variation_diversity')
            
            # Most common variations (binary features)
            common_variations = [
                'linear', 'sinusoidal', 'spherical', 'swirl', 'horseshoe',
                'polar', 'heart', 'disc', 'spiral', 'julia'
            ]
            
            for var_name in common_variations:
                has_variation = any(
                    var_name in xform.get('variations', {})
                    for xform in xforms
                )
                features.append(1.0 if has_variation else 0.0)
                feature_names.append(f'has_{var_name}')
        
        # Color palette features
        if 'colors' in fractal_data and fractal_data['colors']:
            colors = fractal_data['colors']
            
            # Extract RGB statistics from palette
            rgb_values = []
            for color in colors[:50]:  # Sample first 50 colors
                rgb_str = color.get('rgb', '0 0 0')
                rgb_parts = rgb_str.split()
                if len(rgb_parts) >= 3:
                    try:
                        rgb = [float(part) / 255.0 for part in rgb_parts[:3]]
                        rgb_values.extend(rgb)
                    except ValueError:
                        rgb_values.extend([0.0, 0.0, 0.0])
            
            if rgb_values:
                features.extend([
                    np.mean(rgb_values),
                    np.std(rgb_values),
                    np.min(rgb_values),
                    np.max(rgb_values)
                ])
                feature_names.extend(['palette_mean', 'palette_std', 'palette_min', 'palette_max'])
        
        # Store feature names for first extraction
        if not self.feature_names:
            self.feature_names = feature_names
        
        return np.array(features)


class UserPreferenceDataset:
    """Manage user selection data for neural network training."""
    
    def __init__(self, data_dir: str = "data/user_selections"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.feature_extractor = FractalFeatureExtractor()
        self.logger = logging.getLogger(__name__)
    
    def load_selection_data(self) -> List[Dict[str, Any]]:
        """Load all user selection data from JSON files."""
        selection_files = list(self.data_dir.glob("selection_*.json"))
        all_data = []
        
        for file_path in selection_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    all_data.append(data)
            except Exception as e:
                self.logger.error(f"Failed to load {file_path}: {e}")
        
        self.logger.info(f"Loaded {len(all_data)} selection records from {len(selection_files)} files")
        return all_data
    
    def create_training_dataset(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Create training dataset from user selection data.
        
        Returns:
            Tuple of (features, labels, feature_names)
        """
        selection_data = self.load_selection_data()
        
        if not selection_data:
            self.logger.warning("No selection data available for training")
            return np.array([]), np.array([]), []
        
        features_list = []
        labels_list = []
        
        for session in selection_data:
            if session.get('selected_index', -1) == -1:
                continue  # Skip sessions where no selection was made
            
            selected_idx = session['selected_index']
            fractal_params = session.get('fractal_parameters', [])
            
            # Create training examples: selected fractal = positive, others = negative
            for i, fractal_param in enumerate(fractal_params):
                if not fractal_param:  # Skip empty parameter sets
                    continue
                
                try:
                    # Extract features
                    features = self.feature_extractor.extract_flam3_features(fractal_param)
                    
                    if len(features) > 0:
                        features_list.append(features)
                        labels_list.append(1 if i == selected_idx else 0)  # 1 = preferred, 0 = not preferred
                
                except Exception as e:
                    self.logger.error(f"Failed to extract features from fractal {i}: {e}")
                    continue
        
        if not features_list:
            self.logger.warning("No valid features extracted")
            return np.array([]), np.array([]), []
        
        features_array = np.array(features_list)
        labels_array = np.array(labels_list)
        
        self.logger.info(f"Created dataset: {len(features_array)} samples, {features_array.shape[1]} features")
        self.logger.info(f"Label distribution: {np.bincount(labels_array)}")
        
        return features_array, labels_array, self.feature_extractor.feature_names


class FractalPreferencePredictor:
    """Train and use models to predict user fractal preferences."""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        
        self.logger = logging.getLogger(__name__)
    
    def train(self, features: np.ndarray, labels: np.ndarray, feature_names: List[str]):
        """Train the preference prediction model."""
        if len(features) == 0 or len(labels) == 0:
            self.logger.error("Cannot train with empty dataset")
            return
        
        self.feature_names = feature_names
        
        # Normalize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Split data
        if len(features) > 10:  # Only split if we have enough data
            X_train, X_test, y_train, y_test = train_test_split(
                features_scaled, labels, test_size=0.2, random_state=42, stratify=labels
            )
        else:
            X_train, X_test, y_train, y_test = features_scaled, features_scaled, labels, labels
        
        # Train Random Forest model (good baseline)
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'  # Handle class imbalance
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.logger.info(f"Model trained with accuracy: {accuracy:.3f}")
        self.logger.info(f"Classification report:\n{classification_report(y_test, y_pred)}")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            self.logger.info(f"Top 10 important features:\n{importance_df.head(10)}")
        
        self.is_trained = True
    
    def predict_preference(self, fractal_features: np.ndarray) -> float:
        """Predict preference score for a fractal."""
        if not self.is_trained or self.model is None:
            return 0.5  # Neutral score if not trained
        
        try:
            # Ensure features have correct shape
            if len(fractal_features.shape) == 1:
                fractal_features = fractal_features.reshape(1, -1)
            
            # Normalize features
            features_scaled = self.scaler.transform(fractal_features)
            
            # Get probability of being preferred
            prob = self.model.predict_proba(features_scaled)[0][1]  # Probability of class 1 (preferred)
            
            return float(prob)
            
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return 0.5
    
    def save_model(self, filename: str = "fractal_preference_model.pkl"):
        """Save trained model to disk."""
        if not self.is_trained:
            self.logger.warning("No trained model to save")
            return
        
        model_path = self.model_dir / filename
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'training_time': time.time()
        }
        
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            self.logger.info(f"Model saved to {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
    
    def load_model(self, filename: str = "fractal_preference_model.pkl") -> bool:
        """Load trained model from disk."""
        model_path = self.model_dir / filename
        
        if not model_path.exists():
            self.logger.warning(f"Model file {model_path} not found")
            return False
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.is_trained = True
            
            self.logger.info(f"Model loaded from {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def create_ai_selector(self, name: str, description: str = "", 
                          selector_dir: str = "selectors") -> str:
        """Create an AI selector plugin from the trained model."""
        if not self.is_trained:
            self.logger.error("Cannot create AI selector from untrained model")
            return ""
        
        selector_path = Path(selector_dir)
        selector_path.mkdir(parents=True, exist_ok=True)
        
        # Create selector metadata
        timestamp = datetime.now().isoformat()
        selector_data = {
            "name": name,
            "description": description,
            "created_at": timestamp,
            "version": "1.0",
            "type": "preference_predictor",
            "model_type": type(self.model).__name__,
            "feature_count": len(self.feature_names),
            "feature_names": self.feature_names.copy()
        }
        
        # Create selector file
        selector_filename = f"{name.lower().replace(' ', '_')}_selector.pkl"
        selector_file_path = selector_path / selector_filename
        
        full_selector_data = {
            "metadata": selector_data,
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names
        }
        
        try:
            with open(selector_file_path, 'wb') as f:
                pickle.dump(full_selector_data, f)
            
            self.logger.info(f"AI selector created: {selector_file_path}")
            return str(selector_file_path)
            
        except Exception as e:
            self.logger.error(f"Failed to create AI selector: {e}")
            return ""


class AISelector:
    """Load and use AI selector plugins for automated fractal selection."""
    
    def __init__(self, selector_path: str):
        self.selector_path = Path(selector_path)
        self.metadata = None
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.feature_extractor = FractalFeatureExtractor()
        self.logger = logging.getLogger(__name__)
        
        if not self.load_selector():
            raise ValueError(f"Failed to load AI selector from {selector_path}")
    
    def load_selector(self) -> bool:
        """Load AI selector from file."""
        if not self.selector_path.exists():
            self.logger.error(f"Selector file not found: {self.selector_path}")
            return False
        
        try:
            with open(self.selector_path, 'rb') as f:
                selector_data = pickle.load(f)
            
            self.metadata = selector_data['metadata']
            self.model = selector_data['model']
            self.scaler = selector_data['scaler']
            self.feature_names = selector_data['feature_names']
            
            self.logger.info(f"Loaded AI selector: {self.metadata['name']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load selector: {e}")
            return False
    
    def select_fractal(self, fractal_candidates: List[Dict[str, Any]]) -> int:
        """Select the best fractal from candidates using AI prediction."""
        if not fractal_candidates:
            return -1
        
        best_score = -1
        best_index = 0
        
        for i, candidate in enumerate(fractal_candidates):
            try:
                # Extract features
                features = self.feature_extractor.extract_flam3_features(candidate)
                
                if len(features) > 0:
                    # Ensure features have correct shape
                    features = features.reshape(1, -1)
                    
                    # Normalize features
                    features_scaled = self.scaler.transform(features)
                    
                    # Get preference probability
                    prob = self.model.predict_proba(features_scaled)[0][1]
                    
                    if prob > best_score:
                        best_score = prob
                        best_index = i
                        
            except Exception as e:
                self.logger.error(f"Error evaluating candidate {i}: {e}")
                continue
        
        self.logger.info(f"AI selected fractal {best_index} with score {best_score:.3f}")
        return best_index
    
    def get_info(self) -> Dict[str, Any]:
        """Get selector metadata and info."""
        return self.metadata.copy() if self.metadata else {}


class SelectionDataManager:
    """Manage and analyze user selection datasets."""
    
    def __init__(self, data_dir: str = "data/user_selections"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all selection sessions with metadata."""
        session_files = list(self.data_dir.glob("selection_*.json"))
        sessions = []
        
        for file_path in session_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Extract session info
                session_info = {
                    "filename": file_path.name,
                    "filepath": str(file_path),
                    "timestamp": data.get('timestamp', 'unknown'),
                    "generation": data.get('generation', 'unknown'),
                    "selected_index": data.get('selected_index', -1),
                    "num_candidates": len(data.get('fractal_parameters', [])),
                    "selection_time": data.get('selection_time', 0),
                    "file_size": file_path.stat().st_size
                }
                sessions.append(session_info)
                
            except Exception as e:
                self.logger.error(f"Failed to read session {file_path}: {e}")
        
        # Sort by timestamp
        sessions.sort(key=lambda x: x['timestamp'], reverse=True)
        return sessions
    
    def get_selection_stats(self) -> Dict[str, Any]:
        """Get statistics about selection data."""
        sessions = self.list_sessions()
        
        if not sessions:
            return {"total_sessions": 0, "message": "No selection data found"}
        
        total_sessions = len(sessions)
        total_selections = sum(1 for s in sessions if s['selected_index'] != -1)
        avg_selection_time = np.mean([s['selection_time'] for s in sessions if s['selection_time'] > 0])
        total_candidates = sum(s['num_candidates'] for s in sessions)
        
        # Date range
        timestamps = [s['timestamp'] for s in sessions if s['timestamp'] != 'unknown']
        date_range = f"{min(timestamps)} to {max(timestamps)}" if timestamps else "Unknown"
        
        return {
            "total_sessions": total_sessions,
            "total_selections": total_selections,
            "selection_rate": total_selections / total_sessions if total_sessions > 0 else 0,
            "avg_selection_time": avg_selection_time,
            "total_candidates_evaluated": total_candidates,
            "date_range": date_range,
            "data_size_mb": sum(s['file_size'] for s in sessions) / (1024 * 1024)
        }
    
    def clean_invalid_sessions(self) -> int:
        """Remove sessions with invalid or incomplete data."""
        sessions = self.list_sessions()
        removed_count = 0
        
        for session in sessions:
            try:
                with open(session['filepath'], 'r') as f:
                    data = json.load(f)
                
                # Check if session has required data
                is_valid = (
                    'fractal_parameters' in data and
                    'selected_index' in data and
                    len(data['fractal_parameters']) > 0 and
                    data['selected_index'] >= -1
                )
                
                if not is_valid:
                    Path(session['filepath']).unlink()
                    removed_count += 1
                    self.logger.info(f"Removed invalid session: {session['filename']}")
                    
            except Exception as e:
                self.logger.error(f"Error checking session {session['filename']}: {e}")
        
        return removed_count
    
    def export_dataset(self, output_file: str, filter_generations: List[int] = None) -> bool:
        """Export selection data as a single consolidated file."""
        sessions = self.list_sessions()
        
        if filter_generations:
            sessions = [s for s in sessions if s['generation'] in filter_generations]
        
        consolidated_data = {
            "export_timestamp": datetime.now().isoformat(),
            "total_sessions": len(sessions),
            "sessions": []
        }
        
        for session_info in sessions:
            try:
                with open(session_info['filepath'], 'r') as f:
                    session_data = json.load(f)
                consolidated_data["sessions"].append(session_data)
            except Exception as e:
                self.logger.error(f"Failed to read session {session_info['filename']}: {e}")
        
        try:
            with open(output_file, 'w') as f:
                json.dump(consolidated_data, f, indent=2)
            self.logger.info(f"Dataset exported to {output_file}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to export dataset: {e}")
            return False


def list_ai_selectors(selector_dir: str = "selectors") -> List[Dict[str, Any]]:
    """List available AI selector plugins."""
    selector_path = Path(selector_dir)
    if not selector_path.exists():
        return []
    
    selectors = []
    for selector_file in selector_path.glob("*_selector.pkl"):
        try:
            with open(selector_file, 'rb') as f:
                selector_data = pickle.load(f)
            
            metadata = selector_data.get('metadata', {})
            metadata['filepath'] = str(selector_file)
            metadata['file_size'] = selector_file.stat().st_size
            selectors.append(metadata)
            
        except Exception as e:
            logging.error(f"Failed to read selector {selector_file}: {e}")
    
    return selectors


def analyze_user_data():
    """Analyze collected user selection data and train initial model."""
    logging.basicConfig(level=logging.INFO)
    
    print("Analyzing user selection data...")
    
    # Load and analyze data
    dataset = UserPreferenceDataset()
    features, labels, feature_names = dataset.create_training_dataset()
    
    if len(features) == 0:
        print("No training data available. Use the interactive selector to collect data first.")
        return
    
    print(f"Dataset: {len(features)} samples, {len(feature_names)} features")
    print(f"Preferred: {np.sum(labels)}, Not preferred: {len(labels) - np.sum(labels)}")
    
    # Train model
    predictor = FractalPreferencePredictor()
    predictor.train(features, labels, feature_names)
    
    # Save model
    predictor.save_model()
    
    # Create AI selector
    timestamp = datetime.now().strftime("%Y-%m-%d")
    selector_name = f"My Preferences {timestamp}"
    selector_description = f"AI trained on {len(features)} selection examples from {timestamp}"
    
    selector_path = predictor.create_ai_selector(selector_name, selector_description)
    
    if selector_path:
        print(f"\nAI selector created: {selector_path}")
        print("This selector can now be used for automated fractal evolution!")
    
    print("\nModel training complete! The AI can now start learning your preferences.")
    print("Keep using the interactive selector to improve the AI's predictions.")


def manage_selection_data(command: str):
    """Command-line interface for managing selection data."""
    logging.basicConfig(level=logging.INFO)
    manager = SelectionDataManager()
    
    if command == "list":
        sessions = manager.list_sessions()
        print(f"\nFound {len(sessions)} selection sessions:")
        print("-" * 80)
        for session in sessions[:10]:  # Show first 10
            print(f"{session['filename']:<25} | Gen: {session['generation']:<3} | "
                  f"Selected: {session['selected_index']:>2} | Candidates: {session['num_candidates']}")
        if len(sessions) > 10:
            print(f"... and {len(sessions) - 10} more sessions")
    
    elif command == "stats":
        stats = manager.get_selection_stats()
        print("\nSelection Data Statistics:")
        print("-" * 40)
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key.replace('_', ' ').title():<25}: {value:.2f}")
            else:
                print(f"{key.replace('_', ' ').title():<25}: {value}")
    
    elif command == "clean":
        removed = manager.clean_invalid_sessions()
        print(f"\nCleaned {removed} invalid session files.")
    
    elif command == "export":
        output_file = f"selection_data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        success = manager.export_dataset(output_file)
        if success:
            print(f"\nDataset exported to: {output_file}")
        else:
            print("\nFailed to export dataset.")
    
    else:
        print("Available commands: list, stats, clean, export")


def list_selectors():
    """List all available AI selectors."""
    selectors = list_ai_selectors()
    
    if not selectors:
        print("No AI selectors found. Train a model first to create selectors.")
        return
    
    print(f"\nFound {len(selectors)} AI selectors:")
    print("-" * 80)
    
    for selector in selectors:
        print(f"Name: {selector.get('name', 'Unknown')}")
        print(f"Description: {selector.get('description', 'No description')}")
        print(f"Created: {selector.get('created_at', 'Unknown')}")
        print(f"Features: {selector.get('feature_count', 'Unknown')}")
        print(f"Model: {selector.get('model_type', 'Unknown')}")
        print(f"File: {selector.get('filepath', 'Unknown')}")
        print("-" * 40)


def export_selector(selector_name: str, output_path: str = None):
    """Export an AI selector to a shareable file."""
    selectors = list_ai_selectors()
    
    matching_selectors = [s for s in selectors if s.get('name', '').lower() == selector_name.lower()]
    
    if not matching_selectors:
        print(f"No selector found with name '{selector_name}'")
        return
    
    # Get source selector path
    source_path = matching_selectors[0].get('filepath', '')
    if not source_path or not Path(source_path).exists():
        print(f"Selector file not found at {source_path}")
        return
    
    # Create output path if not provided
    if not output_path:
        selector_filename = Path(source_path).name
        output_path = f"exported_{selector_filename}"
    
    try:
        # Create a copy of the selector with metadata
        with open(source_path, 'rb') as f:
            selector_data = pickle.load(f)
        
        # Add export metadata
        selector_data['metadata']['exported_at'] = datetime.now().isoformat()
        selector_data['metadata']['original_path'] = source_path
        
        # Save to output path
        with open(output_path, 'wb') as f:
            pickle.dump(selector_data, f)
        
        print(f"\nSelector '{selector_name}' exported to: {output_path}")
        print("You can share this file with others to use your trained AI selector.")
        
    except Exception as e:
        print(f"Error exporting selector: {e}")


def import_selector(selector_path: str, new_name: str = None):
    """Import an AI selector from a shared file."""
    import_path = Path(selector_path)
    
    if not import_path.exists():
        print(f"File not found: {selector_path}")
        return
    
    try:
        # Load selector data
        with open(import_path, 'rb') as f:
            selector_data = pickle.load(f)
        
        if 'metadata' not in selector_data or 'model' not in selector_data:
            print("Invalid selector file format")
            return
        
        # Create selectors directory if needed
        selector_dir = Path("selectors")
        selector_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate new name if provided
        if new_name:
            selector_data['metadata']['name'] = new_name
        
        # Add import metadata
        selector_data['metadata']['imported_at'] = datetime.now().isoformat()
        selector_data['metadata']['import_source'] = str(import_path)
        
        # Create filename based on name
        name = selector_data['metadata']['name']
        selector_filename = f"{name.lower().replace(' ', '_')}_selector.pkl"
        target_path = selector_dir / selector_filename
        
        # Save to selectors directory
        with open(target_path, 'wb') as f:
            pickle.dump(selector_data, f)
        
        print(f"\nSelector imported as: {name}")
        print(f"Saved to: {target_path}")
        
    except Exception as e:
        print(f"Error importing selector: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("FractalGenesis AI Preference Learning System")
        print("\nUsage:")
        print("  python3 ai/preference_learner.py train          # Train model and create AI selector")
        print("  python3 ai/preference_learner.py data <cmd>     # Manage selection data (list/stats/clean/export)")
        print("  python3 ai/preference_learner.py selectors      # List available AI selectors")
        print("  python3 ai/preference_learner.py export <name>  # Export selector for sharing")
        print("  python3 ai/preference_learner.py import <file>  # Import shared selector")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "train":
        analyze_user_data()
    elif command == "data":
        if len(sys.argv) >= 3:
            manage_selection_data(sys.argv[2])
        else:
            manage_selection_data("stats")
    elif command == "selectors":
        list_selectors()
    elif command == "export":
        if len(sys.argv) >= 3:
            export_selector(sys.argv[2])
        else:
            print("Please specify selector name to export")
    elif command == "import":
        if len(sys.argv) >= 3:
            import_selector(sys.argv[2])
        else:
            print("Please specify file path to import")
    else:
        print(f"Unknown command: {command}")
