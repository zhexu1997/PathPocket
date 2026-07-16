"""
Virchow2 Feature Extractor for Pathology Images
Extracts visual features from pathology images using Virchow2 model
"""

import os
import numpy as np
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import logging
import torch
from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked

logger = logging.getLogger(__name__)


class Virchow2FeatureExtractor:
    """Virchow2 feature extractor for pathology images"""
    
    def __init__(
        self,
        model_name: str = "hf-hub:paige-ai/Virchow2",
        model_path: str = None,
        device: str = "cuda",
        batch_size: int = 1,
        **kwargs
    ):
        """
        Initialize Virchow2 feature extractor
        
        Args:
            model_name: HuggingFace model name (default: "hf-hub:paige-ai/Virchow2")
            model_path: Local path to model weights directory (optional, overrides model_name if provided)
            device: Device to run model on ("cuda" or "cpu")
            batch_size: Batch size for feature extraction
            **kwargs: Additional model initialization parameters
        """
        # If model_path is provided, use it; otherwise use model_name
        if model_path:
            self.model_name = model_path
            self.use_local_model = True
        else:
            self.model_name = model_name
            self.use_local_model = False
        
        # Auto-detect device: if CUDA requested but not available, fall back to CPU
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            self.device = "cpu"
        else:
            self.device = device
        self.batch_size = batch_size
        self.model = None
        self.transforms = None
        self.embedding_dim = 2560  # Virchow2 embedding dimension: class_token (1280) + patch_tokens_mean (1280)
        self._initialized = False
        self._model_config = None  # Store model config for transforms
        
    def _load_model(self):
        """Load Virchow2 model"""
        if self._initialized:
            return
            
        try:
            # Load Virchow2 model using timm
            if self.use_local_model:
                logger.info(f"Loading Virchow2 model from local path: {self.model_name}")
                import json
                
                # Check if path exists
                if not os.path.exists(self.model_name):
                    raise FileNotFoundError(f"Model path does not exist: {self.model_name}")
                
                # Try to load model architecture and weights from local path
                try:
                    # First, try to load config.json to get model architecture info
                    config_path = os.path.join(self.model_name, "config.json")
                    self._model_config = None
                    if os.path.exists(config_path):
                        with open(config_path, 'r') as f:
                            self._model_config = json.load(f)
                        logger.info(f"Loaded model config from: {config_path}")
                    
                    # Try to create model architecture without accessing HuggingFace
                    # Use timm's local model loading capability
                    # Try using the model name without hf-hub prefix, or use a direct architecture name
                    try:
                        # Method 1: Try to use timm's local path support (if available)
                        # Some timm versions support local:// prefix
                        local_model_name = f"local://{self.model_name}"
                        logger.debug(f"Attempting to load model with local path: {local_model_name}")
                        self.model = timm.create_model(
                            local_model_name,
                            pretrained=True,  # This should load from local path
                            mlp_layer=SwiGLUPacked,
                            act_layer=torch.nn.SiLU
                        )
                        logger.info("Successfully loaded model using timm local path support")
                    except Exception as local_error:
                        logger.debug(f"Local path method failed: {local_error}, trying manual loading...")
                        
                        # Method 2: Manually create architecture and load weights
                        logger.info("Attempting to load model architecture and weights manually...")
                        
                        # Find weights file first
                        weights_path = None
                        possible_weight_files = [
                            "pytorch_model.bin",
                            "model.bin", 
                            "model.safetensors",
                        ]
                        
                        for weight_file in possible_weight_files:
                            test_path = os.path.join(self.model_name, weight_file)
                            if os.path.exists(test_path):
                                weights_path = test_path
                                break
                        
                        if not weights_path:
                            import glob
                            safetensors_files = glob.glob(os.path.join(self.model_name, "*.safetensors"))
                            if safetensors_files:
                                weights_path = safetensors_files[0]
                        
                        if not weights_path:
                            raise FileNotFoundError(f"No weight file found in {self.model_name}")
                        
                        # Load state dict
                        logger.info(f"Loading weights from: {weights_path}")
                        if weights_path.endswith('.safetensors'):
                            try:
                                from safetensors.torch import load_file
                                state_dict = load_file(weights_path)
                            except ImportError:
                                raise ImportError("safetensors package required for .safetensors files. Install with: pip install safetensors")
                        else:
                            state_dict = torch.load(weights_path, map_location='cpu')
                        
                        # Create model architecture based on config.json
                        logger.info("Creating model architecture from config...")
                        architecture_name = None
                        model_args = {}
                        
                        if self._model_config:
                            architecture_name = self._model_config.get("architecture")
                            model_args = self._model_config.get("model_args", {})
                            logger.info(f"Using architecture from config: {architecture_name}")
                            logger.info(f"Model args: {model_args}")
                        
                        if not architecture_name:
                            # Fallback: try vit_huge_patch14_224 (Virchow2's architecture)
                            architecture_name = "vit_huge_patch14_224"
                            logger.warning(f"No architecture in config, using fallback: {architecture_name}")
                        
                        try:
                            # Create model with architecture from config
                            create_kwargs = {
                                "pretrained": False,
                                "num_classes": model_args.get("num_classes", 0),
                                "mlp_layer": SwiGLUPacked,
                                "act_layer": torch.nn.SiLU,
                            }
                            
                            # Add model-specific args if available
                            if "img_size" in model_args:
                                create_kwargs["img_size"] = model_args["img_size"]
                            if "init_values" in model_args:
                                create_kwargs["init_values"] = model_args["init_values"]
                            if "reg_tokens" in model_args:
                                create_kwargs["reg_tokens"] = model_args["reg_tokens"]
                            if "mlp_ratio" in model_args:
                                create_kwargs["mlp_ratio"] = model_args["mlp_ratio"]
                            if "global_pool" in model_args:
                                create_kwargs["global_pool"] = model_args["global_pool"]
                            
                            logger.debug(f"Creating model with kwargs: {create_kwargs}")
                            self.model = timm.create_model(architecture_name, **create_kwargs)
                            logger.info(f"Created model architecture: {architecture_name}")
                        except Exception as arch_error:
                            logger.error(f"Failed to create model architecture {architecture_name}: {arch_error}")
                            import traceback
                            logger.error(traceback.format_exc())
                            raise
                        
                        # Load state dict
                        logger.info("Loading state dict...")
                        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                        if missing_keys:
                            logger.warning(f"Missing keys when loading model (first 10): {missing_keys[:10]}")
                        if unexpected_keys:
                            logger.warning(f"Unexpected keys when loading model (first 10): {unexpected_keys[:10]}")
                        logger.info(f"Successfully loaded model weights from {weights_path}")
                        
                except Exception as e:
                    logger.error(f"Failed to load local model: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    raise
            else:
                logger.info(f"Loading Virchow2 model from HuggingFace: {self.model_name}")
                self.model = timm.create_model(
                    self.model_name,
                    pretrained=True,
                    num_classes=0,  # No classification head
                    global_pool="",  # No global pooling, return all tokens
                    mlp_layer=SwiGLUPacked,
                    act_layer=torch.nn.SiLU
                )
            
            self.model = self.model.eval()
            
            # Move model to device, with error handling for CUDA
            try:
                self.model = self.model.to(self.device)
                logger.info(f"Model moved to device: {self.device}")
            except Exception as device_error:
                if self.device == "cuda":
                    logger.warning(f"Failed to move model to CUDA: {device_error}")
                    logger.info("Falling back to CPU")
                    self.device = "cpu"
                    self.model = self.model.to(self.device)
                else:
                    raise
            
            # Create transforms
            # For local models, try to get data config from model or use config.json
            try:
                if self.use_local_model and self._model_config and "pretrained_cfg" in self._model_config:
                    # Use pretrained_cfg from config.json
                    pretrained_cfg_dict = self._model_config["pretrained_cfg"]
                    # Create a simple object-like structure for resolve_data_config
                    class PretrainedCfg:
                        def __init__(self, cfg_dict):
                            for k, v in cfg_dict.items():
                                setattr(self, k, v)
                    pretrained_cfg = PretrainedCfg(pretrained_cfg_dict)
                    data_config = resolve_data_config(pretrained_cfg, model=self.model)
                else:
                    # Use model's pretrained_cfg (for HuggingFace models)
                    data_config = resolve_data_config(self.model.pretrained_cfg, model=self.model)
                self.transforms = create_transform(**data_config)
            except Exception as transform_error:
                logger.warning(f"Failed to create transforms from config: {transform_error}")
                # Fallback: use standard ImageNet transforms or config.json values
                if self.use_local_model and self._model_config and "pretrained_cfg" in self._model_config:
                    pretrained_cfg_dict = self._model_config["pretrained_cfg"]
                    logger.info("Using transforms from config.json pretrained_cfg")
                    self.transforms = create_transform(
                        input_size=tuple(pretrained_cfg_dict.get("input_size", [3, 224, 224])),
                        mean=tuple(pretrained_cfg_dict.get("mean", [0.485, 0.456, 0.406])),
                        std=tuple(pretrained_cfg_dict.get("std", [0.229, 0.224, 0.225])),
                        interpolation=pretrained_cfg_dict.get("interpolation", "bicubic"),
                        crop_pct=pretrained_cfg_dict.get("crop_pct", 1.0)
                    )
                else:
                    logger.info("Using fallback ImageNet transforms")
                    self.transforms = create_transform(
                        input_size=(3, 224, 224),
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                        interpolation='bicubic',
                        crop_pct=1.0
                    )
            
            self._initialized = True
            logger.info(f"Virchow2 model loaded successfully (embedding_dim={self.embedding_dim}, device={self.device})")
            
        except Exception as e:
            logger.error(f"Failed to load Virchow2 model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    async def extract_features(
        self,
        image_paths: Union[str, List[str]],
        **kwargs
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Extract features from pathology images
        
        Args:
            image_paths: Single image path or list of image paths
            **kwargs: Additional parameters for feature extraction
            
        Returns:
            Feature vectors as numpy array(s)
            - Single image: shape (embedding_dim,)
            - Multiple images: shape (num_images, embedding_dim)
        """
        if not self._initialized:
            self._load_model()
            
        # Convert single path to list
        single_image = isinstance(image_paths, str)
        if single_image:
            image_paths = [image_paths]
            
        # Validate image paths
        valid_paths = []
        for path in image_paths:
            if os.path.exists(path):
                valid_paths.append(path)
            else:
                logger.warning(f"Image path does not exist: {path}")
                
        if not valid_paths:
            raise ValueError("No valid image paths provided")
            
        try:
            # Extract features in batches
            all_features = []
            for i in range(0, len(valid_paths), self.batch_size):
                batch_paths = valid_paths[i:i + self.batch_size]
                batch_features = await self._extract_batch(batch_paths, **kwargs)
                all_features.append(batch_features)
                
            # Concatenate all features
            features = np.concatenate(all_features, axis=0) if len(all_features) > 1 else all_features[0]
            
            # Debug: log feature shape
            logger.debug(f"extract_features: single_image={single_image}, features shape={features.shape}, dtype={features.dtype}")
            
            # Return single array for single image, list for multiple
            if single_image:
                result = features[0] if len(features.shape) > 1 else features
                logger.debug(f"Single image result shape: {result.shape if isinstance(result, np.ndarray) else 'N/A'}")
                return result
            else:
                logger.debug(f"Multiple images result shape: {features.shape}")
                return features
                
        except Exception as e:
            logger.error(f"Error extracting Virchow2 features: {e}")
            raise
    
    async def _extract_batch(
        self,
        image_paths: List[str],
        **kwargs
    ) -> np.ndarray:
        """
        Extract features for a batch of images
        
        Args:
            image_paths: List of image paths
            **kwargs: Additional parameters
            
        Returns:
            Feature vectors as numpy array (batch_size, embedding_dim)
        """
        if not self._initialized:
            self._load_model()
        
        # Load and preprocess images
        images_tensor = []
        for image_path in image_paths:
            try:
                image = Image.open(image_path).convert('RGB')
                image_tensor = self.transforms(image)
                images_tensor.append(image_tensor)
            except Exception as e:
                logger.error(f"Failed to load image {image_path}: {e}")
                # Create a zero tensor as fallback
                images_tensor.append(torch.zeros(3, 224, 224))
        
        # Stack images into batch
        batch_tensor = torch.stack(images_tensor).to(self.device)  # size: batch_size x 3 x 224 x 224
        logger.debug(f"batch_tensor shape: {batch_tensor.shape}, dtype: {batch_tensor.dtype}, device: {batch_tensor.device}")
        
        # Extract features
        with torch.no_grad():
            output = self.model(batch_tensor)  # Expected: batch_size x 261 x 1280
            logger.debug(f"Raw model output shape: {output.shape}, dtype: {output.dtype}, ndim: {output.ndim}")
            
            # Debug: log model output shape
            logger.debug(f"Model output shape: {output.shape}, dtype: {output.dtype}")
            
            # Check output shape
            if output.ndim != 3:
                logger.error(f"Unexpected model output shape: {output.shape}, expected 3D tensor (batch_size, num_tokens, hidden_dim)")
                raise ValueError(f"Model output should be 3D tensor, got {output.ndim}D: {output.shape}")
            
            if output.shape[1] < 261:
                logger.warning(f"Model output has fewer tokens than expected: {output.shape[1]} < 261")
            
            # Extract class token (first token)
            class_token = output[:, 0]  # size: batch_size x 1280
            logger.debug(f"class_token shape: {class_token.shape}")
            
            # Extract patch tokens (skip tokens 1-4 which are register tokens)
            # If output has fewer tokens, adjust accordingly
            if output.shape[1] > 5:
                patch_tokens = output[:, 5:]  # size: batch_size x (num_tokens-5) x 1280
            else:
                logger.error(f"Model output has too few tokens: {output.shape[1]}, cannot extract patch tokens")
                raise ValueError(f"Model output has insufficient tokens: {output.shape[1]}")
            
            logger.debug(f"patch_tokens shape: {patch_tokens.shape}")
            
            # Average pool patch tokens
            patch_tokens_mean = patch_tokens.mean(dim=1)  # size: batch_size x 1280
            logger.debug(f"patch_tokens_mean shape: {patch_tokens_mean.shape}")
            
            # Concatenate class token and average pooled patch tokens
            embeddings = torch.cat([class_token, patch_tokens_mean], dim=-1)  # size: batch_size x 2560
            logger.debug(f"embeddings shape: {embeddings.shape}")
        
        # Convert to numpy and move to CPU
        embeddings_np = embeddings.cpu().numpy()
        
        # Debug: log embedding shape
        logger.debug(f"_extract_batch: input {len(image_paths)} images, output shape={embeddings_np.shape}, dtype={embeddings_np.dtype}, ndim={embeddings_np.ndim}")
        
        # Ensure embeddings_np is 2D (batch_size, embedding_dim)
        if embeddings_np.ndim == 1:
            # Single image, reshape to (1, embedding_dim)
            logger.debug(f"Reshaping 1D array to 2D: {embeddings_np.shape} -> (1, {embeddings_np.shape[0]})")
            embeddings_np = embeddings_np.reshape(1, -1)
        elif embeddings_np.ndim == 0:
            # Scalar, this shouldn't happen
            logger.error(f"Unexpected scalar output from model")
            raise ValueError(f"Model returned scalar instead of vector")
        
        # Verify embedding dimension
        if embeddings_np.shape[1] != self.embedding_dim:
            logger.error(f"_extract_batch: Expected embedding_dim={self.embedding_dim}, but got {embeddings_np.shape[1]}")
            logger.error(f"class_token shape: {class_token.shape}, patch_tokens_mean shape: {patch_tokens_mean.shape}")
            logger.error(f"embeddings shape: {embeddings.shape}, embeddings_np shape: {embeddings_np.shape}")
            raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {embeddings_np.shape[1]}")
        
        logger.debug(f"_extract_batch: Final output shape={embeddings_np.shape}, dtype={embeddings_np.dtype}")
        return embeddings_np
    
    def get_embedding_dim(self) -> int:
        """Get the dimension of extracted features"""
        return self.embedding_dim
    
    def is_available(self) -> bool:
        """Check if Virchow2 model is available"""
        try:
            if not self._initialized:
                self._load_model()
            return self.model is not None
        except Exception:
            return False


class Virchow2FeatureExtractorWrapper:
    """Wrapper for Virchow2 feature extractor to match embedding function interface"""
    
    def __init__(self, virchow2_extractor: Virchow2FeatureExtractor):
        """
        Initialize wrapper
        
        Args:
            virchow2_extractor: Virchow2FeatureExtractor instance
        """
        self.virchow2_extractor = virchow2_extractor
        self._embedding_dim = None
        
    async def func(self, image_paths: List[str]) -> List[np.ndarray]:
        """
        Extract features for a list of image paths (matches embedding function interface)
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of feature vectors (each as numpy array)
        """
        if self._embedding_dim is None:
            self._embedding_dim = self.virchow2_extractor.get_embedding_dim()
            
        features = await self.virchow2_extractor.extract_features(image_paths)
        
        # Debug: log feature format
        logger.debug(f"Virchow2FeatureExtractorWrapper.func: input {len(image_paths)} images, features type={type(features)}, shape={features.shape if isinstance(features, np.ndarray) else 'N/A'}")
        
        # Ensure features is a list of arrays
        if isinstance(features, np.ndarray):
            if len(features.shape) == 1:
                # Single image, single feature vector
                logger.debug(f"Single feature vector shape: {features.shape}, dtype: {features.dtype}")
                return [features]
            else:
                # Multiple images
                result = [features[i] for i in range(features.shape[0])]
                logger.debug(f"Multiple feature vectors: {len(result)} vectors, first shape: {result[0].shape if result else 'N/A'}")
                return result
        elif isinstance(features, list):
            logger.debug(f"Features is already a list: {len(features)} items, first shape: {features[0].shape if features and isinstance(features[0], np.ndarray) else 'N/A'}")
            return features
        else:
            raise ValueError(f"Unexpected feature format: {type(features)}")
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension"""
        if self._embedding_dim is None:
            self._embedding_dim = self.virchow2_extractor.get_embedding_dim()
        return self._embedding_dim
    
    @embedding_dim.setter
    def embedding_dim(self, value: int):
        """Set embedding dimension"""
        self._embedding_dim = value


# Backward compatibility aliases (for existing code that uses CONCH naming)
CONCHFeatureExtractor = Virchow2FeatureExtractor
CONCHFeatureExtractorWrapper = Virchow2FeatureExtractorWrapper
