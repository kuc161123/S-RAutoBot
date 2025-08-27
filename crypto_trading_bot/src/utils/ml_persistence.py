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
from ..db.async_database import async_db

logger = structlog.get_logger(__name__)


class MLPersistenceManager:
    """
    Manages ML model persistence to database
    """
    
    def __init__(self):
        pass  # Use async_db directly, no need for instance
        
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
            existing = await async_db.get_ml_model(model_name)
            
            if existing:
                # Store version before the session closes
                current_version = existing.version if existing.version else 0
                new_version = current_version + 1
                
                # Update existing model
                await async_db.update_ml_model(
                    model_name=model_name,
                    model_data=model_bytes,
                    model_metadata=metadata_json,
                    accuracy=accuracy,
                    training_samples=training_samples,
                    version=new_version
                )
                logger.info(f"Updated ML model {model_name} (version {new_version})")
            else:
                # Create new model
                await async_db.create_ml_model(
                    model_name=model_name,
                    model_data=model_bytes,
                    model_metadata=metadata_json,
                    accuracy=accuracy,
                    training_samples=training_samples,
                    version=1
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
            model_record = await async_db.get_ml_model(model_name)
            
            if not model_record:
                logger.info(f"ML model {model_name} not found - will be created after training")
                return None
            
            # Extract data while record is accessible - handle potential None values
            try:
                model_bytes = model_record.model_data
                version = getattr(model_record, 'version', 1) or 1
                accuracy = getattr(model_record, 'accuracy', None)
                training_samples = getattr(model_record, 'training_samples', 0) or 0
            except Exception as attr_error:
                # If we can't access attributes, the session is detached
                logger.warning(f"Session detached for {model_name}, model will be recreated after training")
                # Return None to trigger model recreation
                return None
            
            # Deserialize model
            model_data = pickle.loads(model_bytes)
            
            accuracy_str = f"{accuracy:.2%}" if accuracy else "N/A"
            logger.info(
                f"Loaded ML model {model_name} "
                f"(version {version}, "
                f"accuracy: {accuracy_str}, "
                f"samples: {training_samples})"
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
            model_record = await async_db.get_ml_model(model_name)
            
            if not model_record:
                return None
            
            # Extract values while record is still accessible
            metadata_json = model_record.model_metadata if model_record.model_metadata else '{}'
            version = model_record.version if model_record.version else 1
            accuracy = model_record.accuracy
            training_samples = model_record.training_samples
            created_at = model_record.created_at.isoformat() if model_record.created_at else None
            updated_at = model_record.updated_at.isoformat() if model_record.updated_at else None
            
            metadata = json.loads(metadata_json)
            metadata.update({
                'version': version,
                'accuracy': accuracy,
                'training_samples': training_samples,
                'created_at': created_at,
                'updated_at': updated_at
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
            models = await async_db.list_ml_models()
            
            result = []
            for model in models:
                # Extract values immediately while model is accessible
                result.append({
                    'name': model.model_name if model.model_name else 'unknown',
                    'version': model.version if model.version else 1,
                    'accuracy': model.accuracy if model.accuracy else 0,
                    'training_samples': model.training_samples if model.training_samples else 0,
                    'updated_at': model.updated_at.isoformat() if model.updated_at else None
                })
            
            return result
            
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
            await async_db.delete_ml_model(model_name)
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