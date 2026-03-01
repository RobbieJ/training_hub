import inspect
import os
from dataclasses import fields
from typing import Any, Dict, Optional, Type

import datasets

from . import Algorithm, Backend, AlgorithmRegistry
from training_hub import utils


class InstructLabTrainingSFTBackend(Backend):
    """InstructLab Training backend for SFT algorithm."""
    
    def execute_training(self, algorithm_params: Dict[str, Any]) -> Any:
        """Execute SFT training using instructlab-training."""
        from instructlab.training import (
            PretrainingConfig,
            TorchrunArgs,
            TrainingArgs,
            run_training,
        )

        # Separate torchrun parameters from training parameters
        torchrun_keys = {'nproc_per_node', 'nnodes', 'node_rank', 'rdzv_id', 'rdzv_endpoint', 'master_addr', 'master_port'}
        
        # Extract torchrun parameters
        torchrun_params = {k: v for k, v in algorithm_params.items() if k in torchrun_keys}

        # Extract training parameters (everything except torchrun params)
        # Note: instructlab-training auto-detects loggers based on mlflow_tracking_uri,
        # wandb_project, and tensorboard_log_dir parameters
        training_params = {k: v for k, v in algorithm_params.items() if k not in torchrun_keys}
        
        # Map training_hub parameter names to instructlab-training parameter names
        if 'max_tokens_per_gpu' in training_params:
            training_params['max_batch_len'] = training_params.pop('max_tokens_per_gpu')

        # AdamW parameter translation
        if 'beta1' in training_params and 'beta2' in training_params:
            training_params['adamw_betas'] = (
                training_params.pop('beta1'),
                training_params.pop('beta2')
            )
        if 'eps' in training_params:
            training_params['adamw_eps'] = training_params.pop('eps')
        if 'weight_decay' in training_params:
            training_params['adamw_weight_decay'] = training_params.pop('weight_decay')

        # Create the pretraining config if it was requested
        block_size = training_params.pop('block_size', None)
        document_column_name = training_params.pop('document_column_name', None)
        is_pretraining = training_params.pop('is_pretraining', None)

        if is_pretraining and block_size is None:
            raise ValueError("block_size is required when is_pretraining=True")

        if is_pretraining:
            pretraining_kwargs: Dict[str, Any] = {}
            if document_column_name is not None:
                pretraining_kwargs['document_column_name'] = document_column_name
            training_params['pretraining_config'] = PretrainingConfig(
                block_size=block_size,
                **pretraining_kwargs,
            )

        # Create TrainingArgs with all provided parameters, letting it handle defaults
        training_args = TrainingArgs(**training_params)
        
        # Set up torchrun arguments with single-node defaults (except nproc_per_node)
        final_torchrun_params = utils.get_torchrun_params(torchrun_params)
        torchrun_args = TorchrunArgs(**final_torchrun_params)

        # Execute training
        return run_training(
            torch_args=torchrun_args,
            train_args=training_args
        )


class MiniTrainerSFTBackend(Backend):
    """MiniTrainer backend for SFT algorithm."""

    def execute_training(self, algorithm_params: Dict[str, Any]) -> Any:
        from mini_trainer import (
            PretrainingConfig,
            TorchrunArgs,
            TrainingArgs,
            TrainingMode,
            run_training,
        )

        renames = {
            'warmup_steps': 'num_warmup_steps',
            'model_path': 'model_name_or_path',
            'num_epochs': 'max_epochs',
            'effective_batch_size': 'batch_size',
            'ckpt_output_dir': 'output_dir',
        }
        algorithm_params = {renames.get(k, k): v for k, v in algorithm_params.items()}

        # Populate logging params from environment variables if not explicitly set
        if not algorithm_params.get('mlflow_tracking_uri'):
            algorithm_params['mlflow_tracking_uri'] = os.environ.get('MLFLOW_TRACKING_URI')
        if not algorithm_params.get('mlflow_experiment_name'):
            algorithm_params['mlflow_experiment_name'] = os.environ.get('MLFLOW_EXPERIMENT_NAME')
        if not algorithm_params.get('mlflow_run_name'):
            algorithm_params['mlflow_run_name'] = os.environ.get('MLFLOW_RUN_NAME')
        if not algorithm_params.get('wandb_project'):
            algorithm_params['wandb_project'] = os.environ.get('WANDB_PROJECT')
        if not algorithm_params.get('wandb_entity'):
            algorithm_params['wandb_entity'] = os.environ.get('WANDB_ENTITY')
        if not algorithm_params.get('wandb_run_name'):
            algorithm_params['wandb_run_name'] = os.environ.get('WANDB_RUN_NAME')

        torchrun_args_fields = {f.name for f in fields(TorchrunArgs)}
        training_args_fields = {f.name for f in fields(TrainingArgs)}

        torchrun_args_pre = {
            k: v for k, v in algorithm_params.items() if k in torchrun_args_fields and v is not None
        }
        torchrun_args_pre = utils.get_torchrun_params(torchrun_args_pre)
        torch_args = TorchrunArgs(**torchrun_args_pre)

        data_output_dir = algorithm_params.get('data_output_dir')
        if data_output_dir is None:
            data_output_dir = os.path.join(algorithm_params['output_dir'], '_internal_data_processing')

        training_ready_data_path = self._process_data(
            data_path=algorithm_params['data_path'],
            model_name_or_path=algorithm_params['model_name_or_path'],
            output_dir=data_output_dir,
            max_seq_len=algorithm_params['max_seq_len'],
            num_cpu_procs=8,
            use_processed_dataset=algorithm_params.get('use_processed_dataset', False),
            unmask_messages=algorithm_params.get('unmask_messages', False),
            is_pretraining=algorithm_params.get('is_pretraining', False),
            document_column_name=algorithm_params.get('document_column_name'),
            trust_remote_code=algorithm_params.get('trust_remote_code'),
        )

        training_args_pre = {
            k: v for k, v in algorithm_params.items() if k in training_args_fields and v is not None
        }
        training_args_pre['data_path'] = training_ready_data_path

        if algorithm_params.get('is_pretraining', False):
            if (block_size := algorithm_params.get('block_size')) is None:
                raise ValueError('block_size is required when is_pretraining=True')
            training_args_pre['pretraining_config'] = PretrainingConfig(block_size=block_size)

        if (
            algorithm_params.get('trust_remote_code') is True
            and 'trust_remote_code' not in training_args_fields
        ):
            raise ValueError(
                "The installed mini-trainer version does not support `trust_remote_code`. "
                "Upgrade mini-trainer to a version with Mistral 3 support."
            )

        if not isinstance(
            train_mode := training_args_pre.get('training_mode', TrainingMode.EPOCH),
            TrainingMode,
        ):
            train_mode = TrainingMode(train_mode)
        training_args_pre['training_mode'] = train_mode
        training_args_pre['osft'] = False

        return run_training(
            torch_args=torch_args,
            train_args=TrainingArgs(**training_args_pre),
        )

    def _process_data(
        self,
        model_name_or_path: str,
        data_path: str,
        output_dir: str,
        max_seq_len: int,
        num_cpu_procs: int,
        unmask_messages: bool,
        use_processed_dataset: bool,
        is_pretraining: bool = False,
        document_column_name: str | None = None,
        trust_remote_code: bool | None = None,
    ) -> str:
        from instructlab.training.data_process import (
            process_documents_for_pretraining,
            process_messages_into_input_ids,
        )

        if use_processed_dataset:
            return data_path

        os.makedirs(output_dir, exist_ok=True)

        if trust_remote_code is True:
            # Required for select models that ship custom configuration/model code.
            os.environ['HF_HUB_TRUST_REMOTE_CODE'] = '1'

        if is_pretraining:
            additional_kwargs: Dict[str, Any] = {}
            if document_column_name is not None:
                additional_kwargs['document_column_name'] = document_column_name
            if (
                trust_remote_code is not None
                and 'trust_remote_code' in inspect.signature(process_documents_for_pretraining).parameters
            ):
                additional_kwargs['trust_remote_code'] = trust_remote_code

            process_documents_for_pretraining(
                data_path=data_path,
                data_output_path=output_dir,
                model_path=model_name_or_path,
                num_cpu_procs=num_cpu_procs,
                **additional_kwargs,
            )
        else:
            processing_data_path = data_path
            if unmask_messages:
                ds = datasets.load_dataset('json', data_files=data_path, split='train')
                ds = ds.map(lambda _: {'unmask': True})
                processing_data_path = os.path.join(output_dir, 'intermediate_data.jsonl')
                ds.to_json(processing_data_path)

            additional_kwargs = {}
            if (
                trust_remote_code is not None
                and 'trust_remote_code' in inspect.signature(process_messages_into_input_ids).parameters
            ):
                additional_kwargs['trust_remote_code'] = trust_remote_code

            process_messages_into_input_ids(
                data_path=processing_data_path,
                data_output_path=output_dir,
                model_path=model_name_or_path,
                max_seq_len=max_seq_len,
                num_cpu_procs=num_cpu_procs,
                **additional_kwargs,
            )

        return os.path.join(output_dir, 'data.jsonl')


class SFTAlgorithm(Algorithm):
    """Supervised Fine-Tuning algorithm."""
    
    def __init__(self, backend: Backend, **kwargs):
        self.backend = backend
        self.config = kwargs
    
    def train(self, 
              model_path: str,
              data_path: str, 
              ckpt_output_dir: str,
              # Training parameters (defaults from TrainingArgs)
              num_epochs: Optional[int] = None,
              effective_batch_size: Optional[int] = None,
              learning_rate: Optional[float] = None,
              max_seq_len: Optional[int] = None,
              max_tokens_per_gpu: Optional[int] = None,
              data_output_dir: Optional[str] = None,
              save_samples: Optional[int] = None,
              warmup_steps: Optional[int] = None,
              accelerate_full_state_at_epoch: Optional[bool] = None,
              checkpoint_at_epoch: Optional[bool] = None,
              is_pretraining: Optional[bool] = None,
              block_size: Optional[int] = None,
              document_column_name: Optional[str] = None,
              trust_remote_code: Optional[bool] = None,
              # AdamW optimizer parameters
              beta1: Optional[float] = None,
              beta2: Optional[float] = None,
              eps: Optional[float] = None,
              weight_decay: Optional[float] = None,
              # Torchrun parameters for multi-node support
              nproc_per_node: Optional[str | int] = None,
              nnodes: Optional[int] = None,
              node_rank: Optional[int] = None,
              rdzv_id: Optional[str | int] = None,
              rdzv_endpoint: Optional[str] = None,
              master_addr: Optional[str] = None,
              master_port: Optional[int] = None,
              # Logging parameters
              wandb_project: Optional[str] = None,
              wandb_entity: Optional[str] = None,
              wandb_run_name: Optional[str] = None,
              tensorboard_log_dir: Optional[str] = None,
              mlflow_tracking_uri: Optional[str] = None,
              mlflow_experiment_name: Optional[str] = None,
              mlflow_run_name: Optional[str] = None,
              **kwargs) -> Any:
        """Execute SFT training.
        
        Args:
            model_path: Path to the model to fine-tune
            data_path: Path to the training data
            ckpt_output_dir: Directory to save checkpoints
            num_epochs: Number of training epochs
            effective_batch_size: Effective batch size for training
            learning_rate: Learning rate for training
            max_seq_len: Maximum sequence length
            max_tokens_per_gpu: Maximum tokens per GPU in a mini-batch (hard-cap for memory to avoid OOMs). Used to automatically calculate mini-batch size and gradient accumulation to maintain the desired effective_batch_size while staying within memory limits.
            data_output_dir: Directory to save processed data
            save_samples: Number of samples to save after training (0 disables saving based on sample count)
            warmup_steps: Number of warmup steps
            accelerate_full_state_at_epoch: Whether to save full state at epoch for automatic checkpoint resumption
            checkpoint_at_epoch: Whether to checkpoint at each epoch
            is_pretraining: Enable document-style continual pretraining mode.
            block_size: Required when `is_pretraining=True`. Token length of each document block.
            document_column_name: Column name containing raw documents when `is_pretraining=True` (defaults to "document").
            trust_remote_code: Enable loading model/config classes that require remote custom code.
            beta1: AdamW optimizer beta1 coefficient (momentum).
            beta2: AdamW optimizer beta2 coefficient (RMSprop).
            eps: AdamW optimizer epsilon for numerical stability.
            weight_decay: AdamW optimizer weight decay coefficient.
            nproc_per_node: Number of processes (GPUs) per node
            nnodes: Total number of nodes
            node_rank: Rank of this node (0 to nnodes-1)
            rdzv_id: Unique job ID for rendezvous
            rdzv_endpoint: Master node endpoint for multi-node training
            master_addr: Master node address for distributed training
            master_port: Master node port for distributed training
            wandb_project: Weights & Biases project name
            wandb_entity: Weights & Biases team/entity name
            wandb_run_name: Weights & Biases run name
            tensorboard_log_dir: Directory for TensorBoard logs
            mlflow_tracking_uri: MLflow tracking server URI
            mlflow_experiment_name: MLflow experiment name
            mlflow_run_name: MLflow run name
            **kwargs: Additional parameters passed to the backend
            
        Returns:
            Training result from the backend
        """
        # Build parameters dict, only including non-None values
        params = {'model_path': model_path, 'data_path': data_path, 'ckpt_output_dir': ckpt_output_dir}
        
        # Add optional parameters if provided
        optional_params = {
            'num_epochs': num_epochs,
            'effective_batch_size': effective_batch_size,
            'learning_rate': learning_rate,
            'max_seq_len': max_seq_len,
            'max_tokens_per_gpu': max_tokens_per_gpu,
            'data_output_dir': data_output_dir,
            'save_samples': save_samples,
            'warmup_steps': warmup_steps,
            'accelerate_full_state_at_epoch': accelerate_full_state_at_epoch,
            'checkpoint_at_epoch': checkpoint_at_epoch,
            'is_pretraining': is_pretraining,
            'block_size': block_size,
            'document_column_name': document_column_name,
            'trust_remote_code': trust_remote_code,
            # AdamW optimizer parameters
            'beta1': beta1,
            'beta2': beta2,
            'eps': eps,
            'weight_decay': weight_decay,
            # Torchrun parameters
            'nproc_per_node': nproc_per_node,
            'nnodes': nnodes,
            'node_rank': node_rank,
            'rdzv_id': rdzv_id,
            'rdzv_endpoint': rdzv_endpoint,
            'master_addr': master_addr,
            'master_port': master_port,
            # Logging parameters
            'wandb_project': wandb_project,
            'wandb_entity': wandb_entity,
            'wandb_run_name': wandb_run_name,
            'tensorboard_log_dir': tensorboard_log_dir,
            'mlflow_tracking_uri': mlflow_tracking_uri,
            'mlflow_experiment_name': mlflow_experiment_name,
            'mlflow_run_name': mlflow_run_name,
        }
        
        # Only add non-None parameters (let TrainingArgs handle defaults)
        for key, value in optional_params.items():
            if value is not None:
                params[key] = value
                
        params.update(kwargs)
        
        return self.backend.execute_training(params)
    
    def get_required_params(self) -> Dict[str, Type]:
        """Return required parameters for SFT."""
        return {
            'model_path': str,
            'data_path': str,
            'ckpt_output_dir': str,
            'num_epochs': int,
            'effective_batch_size': int,
            'learning_rate': float,
            'max_seq_len': int,
            'max_batch_len': int,
        }

    def get_optional_params(self) -> Dict[str, Type]:
        """Return optional parameters for SFT."""
        return {
            'max_tokens_per_gpu': int,
            'data_output_dir': str,
            'save_samples': int,
            'warmup_steps': int,
            'accelerate_full_state_at_epoch': bool,
            'checkpoint_at_epoch': bool,
            'is_pretraining': bool,
            'block_size': int,
            'document_column_name': str,
            'trust_remote_code': bool,
            # AdamW optimizer parameters
            'beta1': float,
            'beta2': float,
            'eps': float,
            'weight_decay': float,
            # Torchrun parameters
            'nproc_per_node': str | int,
            'nnodes': int,
            'node_rank': int,
            'rdzv_id': str | int,
            'rdzv_endpoint': str,
            'master_addr': str,
            'master_port': int,
            # Logging parameters
            'wandb_project': str,
            'wandb_entity': str,
            'wandb_run_name': str,
            'tensorboard_log_dir': str,
            'mlflow_tracking_uri': str,
            'mlflow_experiment_name': str,
            'mlflow_run_name': str,
        }


# Register the algorithm and backend
AlgorithmRegistry.register_algorithm('sft', SFTAlgorithm)
AlgorithmRegistry.register_backend('sft', 'instructlab-training', InstructLabTrainingSFTBackend)
AlgorithmRegistry.register_backend('sft', 'mini-trainer', MiniTrainerSFTBackend)


def sft(model_path: str, 
        data_path: str, 
        ckpt_output_dir: str,
        backend: str = "auto",
        # Training parameters (defaults from TrainingArgs)
        num_epochs: Optional[int] = None,
        effective_batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        max_seq_len: Optional[int] = None,
        max_tokens_per_gpu: Optional[int] = None,
        data_output_dir: Optional[str] = None,
        save_samples: Optional[int] = None,
        warmup_steps: Optional[int] = None,
        accelerate_full_state_at_epoch: Optional[bool] = None,
        checkpoint_at_epoch: Optional[bool] = None,
        is_pretraining: Optional[bool] = None,
        block_size: Optional[int] = None,
        document_column_name: Optional[str] = None,
        trust_remote_code: Optional[bool] = None,
        # AdamW optimizer parameters
        beta1: Optional[float] = None,
        beta2: Optional[float] = None,
        eps: Optional[float] = None,
        weight_decay: Optional[float] = None,
        # Torchrun parameters for multi-node support
        nproc_per_node: Optional[str | int] = None,
        nnodes: Optional[int] = None,
        node_rank: Optional[int] = None,
        rdzv_id: Optional[str | int] = None,
        rdzv_endpoint: Optional[str] = None,
        master_addr: Optional[str] = None,
        master_port: Optional[int] = None,
        # Logging parameters
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        tensorboard_log_dir: Optional[str] = None,
        mlflow_tracking_uri: Optional[str] = None,
        mlflow_experiment_name: Optional[str] = None,
        mlflow_run_name: Optional[str] = None,
        **kwargs) -> Any:
    """Convenience function to run SFT training.
    
        Args:
            model_path: Path to the model to fine-tune
            data_path: Path to the training data
            ckpt_output_dir: Directory to save checkpoints
            backend: Backend implementation to use (default: "auto")
        num_epochs: Number of training epochs
        effective_batch_size: Effective batch size for training
        learning_rate: Learning rate for training
        max_seq_len: Maximum sequence length
        max_tokens_per_gpu: Maximum tokens per GPU in a mini-batch (hard-cap for memory to avoid OOMs). Used to automatically calculate mini-batch size and gradient accumulation to maintain the desired effective_batch_size while staying within memory limits.
        data_output_dir: Directory to save processed data
        save_samples: Number of samples to save after training (0 disables saving based on sample count)
        warmup_steps: Number of warmup steps
        accelerate_full_state_at_epoch: Whether to save full state at epoch for automatic checkpoint resumption
        checkpoint_at_epoch: Whether to checkpoint at each epoch
            is_pretraining: Enable document-style continual pretraining mode.
            block_size: Required when `is_pretraining=True`. Token length of each document block.
            document_column_name: Column name containing raw documents when `is_pretraining=True`.
            trust_remote_code: Enable loading model/config classes that require remote custom code.
            beta1: AdamW optimizer beta1 coefficient (momentum).
        beta2: AdamW optimizer beta2 coefficient (RMSprop).
        eps: AdamW optimizer epsilon for numerical stability.
        weight_decay: AdamW optimizer weight decay coefficient.
        nproc_per_node: Number of processes (GPUs) per node for distributed training
        nnodes: Total number of nodes for distributed training
        node_rank: Rank of this node (0 to nnodes-1) for distributed training
        rdzv_id: Unique job ID for rendezvous in distributed training
        rdzv_endpoint: Master node endpoint for multi-node training
        master_addr: Master node address for distributed training
        master_port: Master node port for distributed training
        wandb_project: Weights & Biases project name
        wandb_entity: Weights & Biases team/entity name
        wandb_run_name: Weights & Biases run name
        tensorboard_log_dir: Directory for TensorBoard logs
        mlflow_tracking_uri: MLflow tracking server URI
        mlflow_experiment_name: MLflow experiment name
        mlflow_run_name: MLflow run name
        **kwargs: Additional parameters passed to the backend
    
    Returns:
        Training result from the backend
    """
    from . import create_algorithm
    
    algorithm = create_algorithm(
        'sft',
        backend,
        model_path_or_architecture=model_path,
        trust_remote_code=trust_remote_code,
    )
    return algorithm.train(
        model_path=model_path,
        data_path=data_path,
        ckpt_output_dir=ckpt_output_dir,
        num_epochs=num_epochs,
        effective_batch_size=effective_batch_size,
        learning_rate=learning_rate,
        max_seq_len=max_seq_len,
        max_tokens_per_gpu=max_tokens_per_gpu,
        data_output_dir=data_output_dir,
        save_samples=save_samples,
        warmup_steps=warmup_steps,
        accelerate_full_state_at_epoch=accelerate_full_state_at_epoch,
        checkpoint_at_epoch=checkpoint_at_epoch,
        is_pretraining=is_pretraining,
        block_size=block_size,
        document_column_name=document_column_name,
        trust_remote_code=trust_remote_code,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=weight_decay,
        nproc_per_node=nproc_per_node,
        nnodes=nnodes,
        node_rank=node_rank,
        rdzv_id=rdzv_id,
        rdzv_endpoint=rdzv_endpoint,
        master_addr=master_addr,
        master_port=master_port,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_run_name=wandb_run_name,
        tensorboard_log_dir=tensorboard_log_dir,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
        mlflow_run_name=mlflow_run_name,
        **kwargs
    )
