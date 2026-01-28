import argparse
import functools
import os
from typing import Any

from chex import PRNGKey
from einops import rearrange
from flax import nnx
from jax import random
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.stages import Wrapped
import jax
import jax.experimental.compilation_cache.compilation_cache
import jax.numpy as jnp
from loguru import logger
import optax
import orbax.checkpoint as ocp

from jaxmsst.bsr_dataset import get_datasets, preprocessing_pipeline
from jaxmsst.configs.loader import load_config
from jaxmsst.models.bs_roformer import BSRoformer
from jaxmsst.profiling import memory_usage_params

jax.experimental.compilation_cache.compilation_cache.set_cache_dir("/tmp/jit_cache")


class Trainer:
    def __init__(self, rng: PRNGKey, hp: Any) -> None:
        self.init_step = 0
        self.optimizer = optax.chain(
            optax.adamw(learning_rate=hp.train.learning_rate, b1=hp.train.betas[0], b2=hp.train.betas[1]),
        )
        init_key, self.train_key = random.split(rng, 2)

        attention_type = getattr(hp.model, "attention", "dot_product")
        flash_min_seq_len = int(getattr(hp.model, "flash_min_seq_len", 0) or 0)
        self.bsr_model = BSRoformer(
            dim=hp.model.dim,
            depth=hp.model.depth,
            stereo=hp.model.stereo,
            time_transformer_depth=hp.model.time_transformer_depth,
            freq_transformer_depth=hp.model.freq_transformer_depth,
            attention_type=attention_type,
            flash_min_seq_len=flash_min_seq_len,
            rngs=nnx.Rngs(init_key),
        )

        n_devices = len(jax.devices())

        if jax.process_index() == 0:
            logger.info(f"Available devices: {jax.devices()}")

        device_mesh = mesh_utils.create_device_mesh((n_devices, 1))

        if jax.process_index() == 0:
            logger.info(f"Device mesh: {device_mesh}")

        options = ocp.CheckpointManagerOptions(max_to_keep=3)
        self.checkpoint_manager = ocp.CheckpointManager(
            hp.log.pth_dir,
            options=options,
        )

        self.mesh = Mesh(device_mesh, axis_names=("data", "model"))
        if jax.process_index() == 0:
            logger.info(f"Mesh: {self.mesh}")

        def get_sharding_for_spec(pspec: PartitionSpec) -> NamedSharding:
            return NamedSharding(self.mesh, pspec)

        x_sharding = get_sharding_for_spec(PartitionSpec("data"))

        self.graphdef, params_state = nnx.split(self.bsr_model, nnx.Param)
        self.params = nnx.to_pure_dict(params_state)
        self.params_sharding = jax.tree_map(lambda _: get_sharding_for_spec(PartitionSpec()), self.params)

        bsr_total_bytes, bsr_total_params = memory_usage_params(self.params)
        if jax.process_index() == 0:
            logger.info(f"BSR Model parameter count: {bsr_total_params} using: {bsr_total_bytes}")
            logger.info("JIT compiling step functions...")

        bsr_step_in_sharding: Any = (
            self.params_sharding,
            x_sharding,
            x_sharding,
            None,
        )
        bsr_step_out_sharding: Any = (
            get_sharding_for_spec(PartitionSpec()),
            self.params_sharding,
        )

        def extract_step(
            params: dict,
            raw_audio: jnp.ndarray,
            target_audio: jnp.ndarray,
            prng_key: jnp.ndarray,
            deterministic: bool,
        ) -> tuple[jnp.ndarray, dict]:
            prng_key, dropout_key = random.split(prng_key)
            model = nnx.merge(self.graphdef, params)
            predict_audio = model(
                raw_audio,
                deterministic=deterministic,
                rngs=nnx.Rngs(dropout=dropout_key),
            )
            predict_audio = predict_audio[..., : target_audio.shape[-1]]

            loss = jnp.mean(jnp.abs(predict_audio - target_audio))
            multi_stft_resolution_loss = 0.0

            for window_size in hp.model.multi_stft_resolutions_window_sizes:
                res_stft_kwargs = dict(
                    nfft=max(window_size, hp.model.stft_n_fft),
                    noverlap=window_size - hp.model.multi_stft_hop_size,
                    nperseg=window_size,
                    boundary=None,
                )

                _, _, recon_Y = jax.scipy.signal.stft(
                    rearrange(predict_audio, "... s t -> (... s) t"), **res_stft_kwargs
                )
                _, _, target_Y = jax.scipy.signal.stft(
                    rearrange(target_audio, "... s t -> (... s) t"), **res_stft_kwargs
                )

                multi_stft_resolution_loss = multi_stft_resolution_loss + jnp.mean(jnp.abs(recon_Y - target_Y))

            l1_loss = jnp.mean(jnp.abs(predict_audio - target_audio))
            total_loss = loss + multi_stft_resolution_loss + l1_loss
            return total_loss, params

        self.bsr_train_step: Wrapped = jax.jit(
            functools.partial(extract_step, deterministic=False),
            in_shardings=bsr_step_in_sharding,
            out_shardings=bsr_step_out_sharding,
        )
        self.flops_for_step = 0

    def save_checkpoint(self, global_step: int):
        if self.params is not None:
            self.checkpoint_manager.save(global_step, args=ocp.args.StandardSave(self.params))

    def restore_checkpoint(self):
        step = self.checkpoint_manager.latest_step()
        self.params = self.checkpoint_manager.restore(step)
        self.init_step = step + 1


def main():
    """Main entry point for the training script."""
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/base.yaml", help="Config File for Train")
    parser.add_argument(
        "--model_config_path",
        type=str,
        default=os.getenv("MODEL_CONFIG_PATH"),
        help="Model config path for model/audio settings (optional)",
    )
    parser.add_argument("--hardware", default="tpu", type=str, help="Hardware Type")
    args = parser.parse_args()
    _main_with_args(args)


def _main_with_args(args):
    if args.hardware == "tpu":
        jax.distributed.initialize()
    hp = load_config(
        args.config,
        model_config_path=args.model_config_path,
    )
    rng = random.PRNGKey(hp.train.seed)
    trainer = Trainer(rng, hp)

    if trainer.checkpoint_manager.latest_step() is not None:
        trainer.restore_checkpoint()

    dataset = get_datasets(hp.data_loader.dataset_path)
    data_iterator = preprocessing_pipeline(
        dataset=dataset,
        global_batch_size=hp.data_loader.global_batch_size,
        global_mesh=trainer.mesh,
        segment_length=hp.data.segment_size,
        grain_worker_count=hp.data_loader.worker_count,
        dataloading_host_index=jax.process_index(),
        dataloading_host_count=hp.data_loader.host_number,
        data_columns=hp.data.data_columns,
        shuffle=hp.data_loader.shuffle,
        data_shuffle_seed=hp.train.seed,
        num_epochs=hp.data_loader.num_epochs,
        drop_remainder=hp.data_loader.drop_remainder,
    )

    for step in range(trainer.init_step, hp.train.total_steps):
        example_batch = next(data_iterator)

        step_key = jax.jit(jax.random.fold_in)(rng, step)

        bsr_train_loss, updated_params = trainer.bsr_train_step(
            trainer.params,
            example_batch["mixture"],
            example_batch["vocals"],
            step_key,
        )
        trainer.params = updated_params

        if step % hp.log.info_interval == 0:
            if jax.process_index() == 0:
                logger.info(f"step: {step} bsr_train_loss: {bsr_train_loss}")

        if step % hp.log.eval_interval == 0:
            trainer.save_checkpoint(step)


if __name__ == "__main__":
    main()
