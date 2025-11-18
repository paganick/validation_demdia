"""Pipeline configuration management for postprocessing.

This module provides utilities for loading and validating pipeline configuration
from YAML files. The configuration controls which steps to run, what outputs to
generate, and how to organize the results.
"""

import os
import yaml
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class StorageConfig:
    """Storage format configuration."""
    use_consolidated_format: bool = True
    keep_legacy_csv: bool = False
    save_comparison_csv: bool = False
    save_feature_cache: bool = False


@dataclass
class ProcessingConfig:
    """Processing steps configuration."""
    selection_methods: List[str] = field(default_factory=lambda: ["random", "ml", "cosine"])
    validation_methods: List[str] = field(default_factory=lambda: ["bert", "features"])
    consolidate_validation: bool = True


@dataclass
class OutputsConfig:
    """Output generation configuration."""
    results_dir: str = "results"
    generate_plots: bool = True
    generate_summary_csv: bool = True
    save_confusion_matrices: bool = True
    save_feature_importance: bool = True
    save_predictions: bool = True


@dataclass
class PerformanceConfig:
    """Performance and caching configuration."""
    n_jobs: Optional[int] = None
    batch_size: int = 32
    use_cache: bool = True
    cache_dir: str = ".cache"


@dataclass
class PipelineConfig:
    """Complete pipeline configuration.

    Attributes:
        storage: Storage format settings
        processing: Processing steps to run
        outputs: Output generation settings
        performance: Performance and caching settings
    """
    storage: StorageConfig = field(default_factory=StorageConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    outputs: OutputsConfig = field(default_factory=OutputsConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'PipelineConfig':
        """Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            PipelineConfig instance with loaded settings

        Raises:
            FileNotFoundError: If yaml_path doesn't exist
            yaml.YAMLError: If YAML is malformed
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
        """Create configuration from dictionary.

        Args:
            config_dict: Dictionary with configuration values

        Returns:
            PipelineConfig instance
        """
        return cls(
            storage=StorageConfig(**config_dict.get('storage', {})),
            processing=ProcessingConfig(**config_dict.get('processing', {})),
            outputs=OutputsConfig(**config_dict.get('outputs', {})),
            performance=PerformanceConfig(**config_dict.get('performance', {}))
        )

    @classmethod
    def default(cls) -> 'PipelineConfig':
        """Create default configuration.

        Returns:
            PipelineConfig with default settings
        """
        return cls()

    def validate(self) -> List[str]:
        """Validate configuration settings.

        Returns:
            List of validation warning/error messages (empty if valid)
        """
        warnings = []

        # Validate selection methods
        valid_methods = {'random', 'ml', 'cosine'}
        invalid = set(self.processing.selection_methods) - valid_methods
        if invalid:
            warnings.append(f"Invalid selection methods: {invalid}")

        # Validate validation methods
        valid_validation = {'bert', 'features', 'empath'}
        invalid = set(self.processing.validation_methods) - valid_validation
        if invalid:
            warnings.append(f"Invalid validation methods: {invalid}")

        # Check batch size
        if self.performance.batch_size <= 0:
            warnings.append("batch_size must be positive")

        # Warn if consolidated format disabled with comparison CSV
        if not self.storage.use_consolidated_format and self.storage.save_comparison_csv:
            warnings.append(
                "save_comparison_csv=True with use_consolidated_format=False "
                "may generate many files"
            )

        return warnings


def load_config(config_path: Optional[str] = None) -> PipelineConfig:
    """Load pipeline configuration from file or use defaults.

    Args:
        config_path: Path to YAML config file. If None, looks for
            'pipeline_config.yaml' in current directory, then uses defaults.

    Returns:
        PipelineConfig instance
    """
    # Try specified path
    if config_path:
        return PipelineConfig.from_yaml(config_path)

    # Try default location
    default_path = 'pipeline_config.yaml'
    if os.path.exists(default_path):
        print(f"Loading config from {default_path}")
        return PipelineConfig.from_yaml(default_path)

    # Use defaults
    print("No config file found, using defaults")
    return PipelineConfig.default()


if __name__ == "__main__":
    # Test configuration loading
    config = load_config()
    warnings = config.validate()

    if warnings:
        print("Configuration warnings:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("✅ Configuration valid")

    print(f"\nStorage: use_consolidated={config.storage.use_consolidated_format}")
    print(f"Selection methods: {config.processing.selection_methods}")
    print(f"Validation methods: {config.processing.validation_methods}")
