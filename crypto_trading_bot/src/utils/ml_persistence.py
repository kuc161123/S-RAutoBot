"""
ML Model Persistence Manager
Handles saving and loading ML models to/from database
"""
import pickle
import json
import base64
from typing import Dict, Any, Optional, List
from datetime import datetime
import structlog
from ..db.models import MLModel
from ..db.database import DatabaseManager

logger = structlog.get_logger(__name__)


class MLPersistenceManager:
    """
    Manages ML model persistence to database
    """
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        
    async def save_model(
        self,
        model_name: str,
        model_data: Any,
        metadata: Dict[str, Any] = None,
        accuracy: float = None,
        training_samples: int = None
    ) -> bool:
        """
        Save ML model to database
        
        Args:
            model_name: Unique identifier for the model
            model_data: The model object to save
            metadata: Additional metadata about the model
            accuracy: Model accuracy score
            training_samples: Number of samples used for training
        
        Returns:
            Success status
        """
        try:
            # Serialize model
            model_bytes = pickle.dumps(model_data)
            
            # Prepare metadata
            if metadata:
                metadata_json = json.dumps(metadata)
            else:
                metadata_json = json.dumps({
                    'saved_at': datetime.utcnow().isoformat(),
                    'model_type': type(model_data).__name__
                })
            
            # Check if model exists
            existing = self.db_manager.get_ml_model(model_name)
            
            if existing:
                # Update existing model
                self.db_manager.update_ml_model(
                    model_name=model_name,
                    model_data=model_bytes,
                    model_metadata=metadata_json,
                    accuracy=accuracy,
                    training_samples=training_samples,
                    version=existing.version + 1
                )
                logger.info(f"Updated ML model {model_name} (version {existing.version + 1})")
            else:
                # Create new model
                self.db_manager.create_ml_model(
                    model_name=model_name,
                    model_data=model_bytes,
                    model_metadata=metadata_json,
                    accuracy=accuracy,
                    training_samples=training_samples
                )
                logger.info(f"Saved new ML model {model_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save ML model {model_name}: {e}")
            return False
    
    async def load_model(self, model_name: str) -> Optional[Any]:
        """
        Load ML model from database
        
        Args:
            model_name: Model identifier
        
        Returns:
            Deserialized model object or None
        """
        try:
            # Get model from database
            model_record = self.db_manager.get_ml_model(model_name)
            
            if not model_record:
                logger.warning(f"ML model {model_name} not found")
                return None
            
            # Deserialize model
            model_data = pickle.loads(model_record.model_data)
            
            accuracy_str = f"{model_record.accuracy:.2%}" if model_record.accuracy else "N/A"
            logger.info(
                f"Loaded ML model {model_name} "
                f"(version {model_record.version}, "
                f"accuracy: {accuracy_str}, "
                f"samples: {model_record.training_samples})"
            )
            
            return model_data
            
        except Exception as e:
            logger.error(f"Failed to load ML model {model_name}: {e}")
            return None
    
    async def get_model_metadata(self, model_name: str) -> Optional[Dict]:
        """
        Get model metadata without loading the full model
        
        Args:
            model_name: Model identifier
        
        Returns:
            Model metadata dictionary or None
        """
        try:
            model_record = self.db_manager.get_ml_model(model_name)
            
            if not model_record:
                return None
            
            metadata = json.loads(model_record.model_metadata)
            metadata.update({
                'version': model_record.version,
                'accuracy': model_record.accuracy,
                'training_samples': model_record.training_samples,
                'created_at': model_record.created_at.isoformat(),
                'updated_at': model_record.updated_at.isoformat()
            })
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to get metadata for {model_name}: {e}")
            return None
    
    async def list_models(self):
        """
        List all available ML models
        
        Returns:
            List of model metadata dictionaries
        """
        try:
            models = self.db_manager.list_ml_models()
            
            return [
                {
                    'name': model.model_name,
                    'version': model.version,
                    'accuracy': model.accuracy,
                    'training_samples': model.training_samples,
                    'updated_at': model.updated_at.isoformat()
                }
                for model in models
            ]
            
        except Exception as e:
            logger.error(f"Failed to list ML models: {e}")
            return []
    
    async def delete_model(self, model_name: str) -> bool:
        """
        Delete ML model from database
        
        Args:
            model_name: Model identifier
        
        Returns:
            Success status
        """
        try:
            self.db_manager.delete_ml_model(model_name)
            logger.info(f"Deleted ML model {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete ML model {model_name}: {e}")
            return False
    
    async def cleanup_old_versions(self, keep_versions: int = 3):
        """
        Clean up old model versions, keeping only the most recent ones
        
        Args:
            keep_versions: Number of versions to keep per model
        """
        try:
            # This would require a more complex query to group by model_name
            # and keep only the most recent versions
            # For now, we'll keep all versions
            logger.info("Model version cleanup not yet implemented")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old model versions: {e}")


# Global instance
ml_persistence = MLPersistenceManager()