"""
Configuration utilities for MCPM
"""

import os
import json
import logging
from typing import Dict, List, Any

# Client detection will be handled by ClientRegistry

logger = logging.getLogger(__name__)

# Default configuration paths
DEFAULT_CONFIG_DIR = os.path.expanduser("~/.config/mcp")
DEFAULT_CONFIG_FILE = os.path.join(DEFAULT_CONFIG_DIR, "config.json")

class ConfigManager:
    """Manages MCP basic configuration
    
    Note: This class now only manages basic system configuration.
    Server configurations are managed by each client independently.
    """
    
    def __init__(self, config_path: str = DEFAULT_CONFIG_FILE):
        self.config_path = config_path
        self.config_dir = os.path.dirname(config_path)
        self._config = None
        self._ensure_dirs()
        self._load_config()
    
    def _ensure_dirs(self) -> None:
        """Ensure all configuration directories exist"""
        os.makedirs(self.config_dir, exist_ok=True)
    
    def _load_config(self) -> None:
        """Load configuration from file or create default"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    self._config = json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Error parsing config file: {self.config_path}")
                self._config = self._default_config()
        else:
            self._config = self._default_config()
            self._save_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Create default configuration"""
        # We'll import here to avoid circular imports
        from mcpm.utils.client_registry import ClientRegistry
        
        # Get recommended client from registry
        recommended_client = ClientRegistry.get_recommended_client()
        installed_clients = ClientRegistry.detect_installed_clients()
        
        return {
            "version": "0.2.0",
            "active_client": recommended_client,
            "clients": {
                "claude-desktop": {
                    "installed": installed_clients.get("claude-desktop", False)
                },
                "cursor": {
                    "installed": installed_clients.get("cursor", False)
                },
                "windsurf": {
                    "installed": installed_clients.get("windsurf", False)
                }
            }
        }
    
    def _save_config(self) -> None:
        """Save current configuration to file"""
        with open(self.config_path, 'w') as f:
            json.dump(self._config, f, indent=2)
    
    def get_config(self) -> Dict[str, Any]:
        """Get the complete configuration"""
        return self._config
    

    
    def _get_client_manager(self, client_name: str):
        """Get the appropriate client manager for a client
        
        Args:
            client_name: Name of the client
            
        Returns:
            BaseClientManager or None if client not supported
        """
        # We'll import here to avoid circular imports
        from mcpm.utils.client_registry import ClientRegistry
        return ClientRegistry.get_client_manager(client_name)
        

        
    def get_active_client(self) -> str:
        """Get the name of the currently active client"""
        return self._config.get("active_client", "claude-desktop")
    
    def set_active_client(self, client_name: str) -> bool:
        """Set the active client"""
        if client_name not in self._config.get("clients", {}):
            logger.error(f"Unknown client: {client_name}")
            return False
        
        self._config["active_client"] = client_name
        self._save_config()
        return True
    
    def get_supported_clients(self) -> List[str]:
        """Get a list of supported client names"""
        # We'll import here to avoid circular imports
        from mcpm.utils.client_registry import ClientRegistry
        return ClientRegistry.get_supported_clients()
