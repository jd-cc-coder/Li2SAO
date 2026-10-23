import abc
import logging
import os
import random
from datetime import timedelta

import ray
import torch
import torch.distributed as dist

import slime.utils.eval_config
from slime.ray.ray_actor import RayActor
from slime.ray.utils import local_rank_for_visible_gpu
from slime.utils.distributed_utils import init_gloo_group
from slime.utils.logging_utils import configure_logger
from slime.utils.memory_utils import clear_memory, print_memory

logger = logging.getLogger(__name__)


def get_local_gpu_id():
    gpu_ids = ray.get_gpu_ids()
    if not gpu_ids:
        raise RuntimeError("Ray actor was scheduled without a GPU (ray.get_gpu_ids() is empty).")
    return local_rank_for_visible_gpu(gpu_ids[0], os.environ.get("CUDA_VISIBLE_DEVICES"))


class TrainRayActor(RayActor):
    def __init__(self, world_size, rank, master_addr, master_port):
        configure_logger()

        self._world_size = world_size
        self._rank = rank
        if master_addr:
            self.master_addr, self.master_port = master_addr, master_port
        else:
            self.master_addr, self.master_port = self._get_current_node_ip_and_free_port(
                start_port=random.randint(20000, 21000)
            )

        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)
        # TODO: currently this doesn't work as ray has already set torch.cuda.device_count().
        # os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        # os.environ["LOCAL_RANK"] = str(ray.get_gpu_ids()[0])
        os.environ["LOCAL_RANK"] = str(get_local_gpu_id())

    def init(self, args, role, with_ref=False, with_opd_teacher=False):
        self.args = args
        self.role = role
        self.with_ref = with_ref
        self.with_opd_teacher = with_opd_teacher

        torch.serialization.add_safe_globals([slime.utils.eval_config.EvalDatasetConfig])

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        nvidia_devs = []
        try:
            nvidia_devs = sorted(name for name in os.listdir("/dev") if name.startswith("nvidia"))
        except OSError:
            pass
        logger.info(
            "CUDA setup: LOCAL_RANK=%s CUDA_VISIBLE_DEVICES=%s NVIDIA_VISIBLE_DEVICES=%s "
            "ray_gpu_ids=%s device_count=%s /dev/nvidia*=%s",
            local_rank,
            os.environ.get("CUDA_VISIBLE_DEVICES"),
            os.environ.get("NVIDIA_VISIBLE_DEVICES"),
            ray.get_gpu_ids(),
            torch.cuda.device_count(),
            nvidia_devs,
        )
        if torch.cuda.device_count() == 0:
            raise RuntimeError(
                "torch sees 0 CUDA devices in this Ray actor even though "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r} "
                f"ray_gpu_ids={ray.get_gpu_ids()} /dev/nvidia*={nvidia_devs} "
                f"NVIDIA_VISIBLE_DEVICES={os.environ.get('NVIDIA_VISIBLE_DEVICES')!r}. "
                "Ray GPU resources are only logical. Recreate the slime container with "
                "`docker run --gpus all --network=host ...` and restart the Ray cluster."
            )
        torch.cuda.set_device(f"cuda:{local_rank}")

        backend = args.distributed_backend

        dist.init_process_group(
            backend=backend,
            timeout=timedelta(minutes=args.distributed_timeout_minutes),
        )
        init_gloo_group()

        args.rank = dist.get_rank()
        args.world_size = dist.get_world_size()

        try:
            if torch.version.hip is not None:
                logger.info("Detected ROCm/HIP environment, skipping NUMA affinity setup")
                # will find the coresponding API to implement ROCm version as below
            else:
                import pynvml

                pynvml.nvmlInit()

                local_rank = int(os.environ["RANK"]) % args.num_gpus_per_node

                handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
                pynvml.nvmlDeviceSetCpuAffinity(handle)

                logger.info(f"Set NUMA affinity for GPU {local_rank}")
                pynvml.nvmlShutdown()

        except ImportError:
            logger.info("Warning: pynvml not available, skipping NUMA affinity setup")
        except Exception as e:
            logger.info(f"Warning: Failed to set NUMA affinity: {e}")

    def clear_memory(self):
        if self.args.debug_rollout_only:
            return
        print_memory("before TrainRayActor.clear_memory")
        clear_memory()
        print_memory("after TrainRayActor.clear_memory")

    @abc.abstractmethod
    def sleep(self, tags):
        raise NotImplementedError

    @abc.abstractmethod
    def wake_up(self, tags):
        raise NotImplementedError

    @abc.abstractmethod
    def train(self, rollout_id, rollout_data_ref, external_data=None):
        raise NotImplementedError

    @abc.abstractmethod
    def save_model(self, rollout_id, force_sync=False):
        raise NotImplementedError

    @abc.abstractmethod
    def update_weights(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _get_parallel_config(self):
        raise NotImplementedError

    def set_rollout_manager(self, rollout_manager):
        self.rollout_manager = rollout_manager
        if not self.args.debug_rollout_only and self.args.rank == 0:
            ray.get(self.rollout_manager.set_train_parallel_config.remote(self.train_parallel_config))

    def update_args_fields(self, fields: dict) -> None:
        """Mutate actor (and in-process critic) args in place after a Li²SAO switch."""
        targets = [self.args]
        critic_args = getattr(self, "critic_args", None)
        if critic_args is not None and critic_args is not self.args:
            targets.append(critic_args)
        for obj in targets:
            if obj is None:
                continue
            for key, value in fields.items():
                setattr(obj, key, value)
