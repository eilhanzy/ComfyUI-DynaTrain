from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


class FakeTensor:
    def __init__(self, shape):
        self.shape = shape


def load_package():
    spec = importlib.util.spec_from_file_location(
        "dynatrain",
        REPO_ROOT / "__init__.py",
        submodule_search_locations=[str(REPO_ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class DynaTrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.package = load_package()
        from dynatrain.nodes import PlotLossGraphNode, SaveLoRAWeightsNode, TrainLoRAAdvancedNode, TrainLoRANode
        from dynatrain.trainers.lora_train_advanced import (
            build_lora_payload,
            build_loss_map,
            get_job_status,
            prepare_training_run,
        )
        from dynatrain.utils.runtime import loss_history
        from dynatrain.trainers.precision_planner import plan_precision
        from dynatrain.trainers.preview_runner import build_preview_config
        from dynatrain.validators.caption_pairs import validate_caption_pairs
        from dynatrain.validators.dataset_sanity import summarize_dataset

        cls.PlotLossGraphNode = PlotLossGraphNode
        cls.SaveLoRAWeightsNode = SaveLoRAWeightsNode
        cls.TrainLoRAAdvancedNode = TrainLoRAAdvancedNode
        cls.TrainLoRANode = TrainLoRANode
        cls.build_lora_payload = staticmethod(build_lora_payload)
        cls.build_loss_map = staticmethod(build_loss_map)
        cls.get_job_status = staticmethod(get_job_status)
        cls.loss_history = staticmethod(loss_history)
        cls.prepare_training_run = staticmethod(prepare_training_run)
        cls.plan_precision = staticmethod(plan_precision)
        cls.build_preview_config = staticmethod(build_preview_config)
        cls.validate_caption_pairs = staticmethod(validate_caption_pairs)
        cls.summarize_dataset = staticmethod(summarize_dataset)

    def test_import_exposes_all_nodes(self):
        names = self.package.NODE_CLASS_MAPPINGS
        self.assertIn("DynaTrainTrainingJobStatus", names)
        self.assertIn("DynaTrainTrainLoRAAdvanced", names)
        self.assertIn("DynaTrainTrainLoRA", names)
        self.assertIn("DynaTrainSaveLoRAWeights", names)
        self.assertIn("DynaTrainPlotLossGraph", names)

    def test_dataset_empty_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = self.summarize_dataset(tmp)
        self.assertTrue(report["blocked"])
        self.assertIn("Dataset contains no supported image files.", report["errors"])

    def test_dataset_single_image_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "only.jpg").write_bytes(b"image")
            report = self.summarize_dataset(tmp)
        self.assertTrue(report["blocked"])
        self.assertIn("only one image", " ".join(report["errors"]).lower())

    def test_missing_and_empty_caption_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.jpg").write_bytes(b"image-a")
            (root / "b.jpg").write_bytes(b"image-b")
            (root / "a.txt").write_text("", encoding="utf-8")
            report = self.summarize_dataset(tmp)
            validated = self.validate_caption_pairs(dataset_report=report)
        self.assertTrue(validated["blocked"])
        self.assertEqual(validated["missing_captions"], ["b.jpg"])
        self.assertEqual(validated["empty_captions"], ["a.jpg"])

    def test_duplicate_and_repeated_caption_warnings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.jpg").write_bytes(b"same-image")
            (root / "b.jpg").write_bytes(b"same-image")
            (root / "a.txt").write_text("same caption", encoding="utf-8")
            (root / "b.txt").write_text("same caption", encoding="utf-8")
            report = self.summarize_dataset(tmp)
            validated = self.validate_caption_pairs(dataset_report=report, warn_repeated_caption_ratio=0.2)
        self.assertFalse(validated["blocked"])
        joined = "\n".join(validated["warnings"])
        self.assertIn("Duplicate-heavy dataset", joined)
        self.assertIn("Repeated captions exceed the warning threshold", joined)

    def test_precision_planner_fallbacks_and_warnings(self):
        low_vram = self.plan_precision(
            model_family="sd15",
            vram_gb=6.0,
            requested_precision="auto",
            save_precision="auto",
            optimizer="adamw8bit",
            gradient_accumulation_steps=2,
            gpu_supports_bf16=True,
            allow_experimental_fp8=False,
        )
        self.assertEqual(low_vram["train_precision"], "fp16")

        bf16_fallback = self.plan_precision(
            model_family="sd15",
            vram_gb=24.0,
            requested_precision="bf16",
            save_precision="auto",
            optimizer="adamw8bit",
            gradient_accumulation_steps=2,
            gpu_supports_bf16=False,
            allow_experimental_fp8=False,
        )
        self.assertEqual(bf16_fallback["train_precision"], "fp16")

        fp8_guard = self.plan_precision(
            model_family="sdxl",
            vram_gb=24.0,
            requested_precision="fp8",
            save_precision="auto",
            optimizer="lion",
            gradient_accumulation_steps=13,
            gpu_supports_bf16=True,
            allow_experimental_fp8=False,
        )
        self.assertEqual(fp8_guard["train_precision"], "fp16")
        self.assertIn("recommended safety limit", "\n".join(fp8_guard["warnings"]))

        flux_plan = self.plan_precision(
            model_family="flux",
            vram_gb=16.0,
            requested_precision="nvfp4",
            save_precision="bf16",
            optimizer="adamw8bit",
            gradient_accumulation_steps=7,
            gpu_supports_bf16=True,
            allow_experimental_fp8=True,
        )
        self.assertEqual(flux_plan["preset"]["resolution"], 1024)
        self.assertIn("NVFP4", "\n".join(flux_plan["warnings"]))
        self.assertIn("recommended safety limit", "\n".join(flux_plan["warnings"]))

    def test_dry_run_generates_module_and_script_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._make_validated_dataset(tmp)
            precision = self._make_precision()
            preview = self.build_preview_config(
                enabled=True,
                sample_every_n_steps=25,
                sample_prompts="portrait photo",
                sampler="euler_a",
                sample_steps=20,
                cfg_scale=7.0,
                seed=42,
            )
            runtime_root = str(Path(tmp) / "runtime")
            output_dir = str(Path(tmp) / "out")
            module_run = self.prepare_training_run(
                validated_dataset=dataset,
                precision_plan=precision,
                preview_config=preview,
                base_model_path="/models/base.safetensors",
                output_dir=output_dir,
                output_name="module_case",
                backend_workdir=str(REPO_ROOT),
                python_executable=sys.executable,
                entrypoint_mode="module",
                module_name="tests.fake_onetrainer_backend",
                script_path="",
                network_dim=16,
                network_alpha=16,
                learning_rate=1e-4,
                max_train_steps=100,
                lr_scheduler="cosine",
                resolution_x=768,
                resolution_y=1024,
                batch_size_override=0,
                gradient_accumulation_override=0,
                execute=False,
                extra_args="",
                runtime_root=runtime_root,
            )
            script_run = self.prepare_training_run(
                validated_dataset=dataset,
                precision_plan=precision,
                preview_config=preview,
                base_model_path="/models/base.safetensors",
                output_dir=output_dir,
                output_name="script_case",
                backend_workdir=str(REPO_ROOT),
                python_executable=sys.executable,
                entrypoint_mode="script",
                module_name="",
                script_path=str(REPO_ROOT / "tests" / "fake_onetrainer_backend.py"),
                network_dim=16,
                network_alpha=16,
                learning_rate=1e-4,
                max_train_steps=100,
                lr_scheduler="cosine",
                resolution_x=512,
                resolution_y=768,
                batch_size_override=0,
                gradient_accumulation_override=0,
                execute=False,
                extra_args="",
                runtime_root=runtime_root,
            )
        self.assertEqual(module_run["status"], "dry_run")
        self.assertTrue(module_run["backend_ready"])
        self.assertIn("importable", module_run["backend_check_message"])
        self.assertEqual(module_run["command_list"][:3], [sys.executable, "-m", "tests.fake_onetrainer_backend"])
        self.assertEqual(module_run["config_payload"]["training"]["resolution_x"], 768)
        self.assertEqual(module_run["config_payload"]["training"]["resolution_y"], 1024)
        self.assertTrue(script_run["backend_ready"])
        self.assertEqual(script_run["command_list"][:2], [sys.executable, str(REPO_ROOT / "tests" / "fake_onetrainer_backend.py")])
        self.assertEqual(script_run["config_payload"]["training"]["resolution_x"], 512)
        self.assertEqual(script_run["config_payload"]["training"]["resolution_y"], 768)
        self.assertTrue(module_run["config_path"].endswith("_onetrainer_config.json"))

    def test_preview_config_tracks_comfy_connections(self):
        preview = self.build_preview_config(
            enabled=True,
            sample_every_n_steps=25,
            sample_prompts="portrait photo",
            sampler="euler_a",
            sample_steps=20,
            cfg_scale=7.0,
            seed=42,
            model=object(),
            positive=[["cond", {"pooled_output": None}]],
            latents={"samples": FakeTensor((1, 4, 64, 64))},
        )
        self.assertTrue(preview["uses_comfy_preview_inputs"])
        self.assertTrue(preview["connected_inputs"]["model"]["connected"])
        self.assertEqual(preview["connected_inputs"]["latents"]["shape"], (1, 4, 64, 64))

    def test_train_job_carries_comfy_connections(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._make_validated_dataset(tmp)
            precision = self._make_precision()
            preview = self.build_preview_config(
                enabled=True,
                sample_every_n_steps=25,
                sample_prompts="portrait photo",
                sampler="euler_a",
                sample_steps=20,
                cfg_scale=7.0,
                seed=42,
            )
            job = self.prepare_training_run(
                validated_dataset=dataset,
                precision_plan=precision,
                preview_config=preview,
                base_model_path="/models/base.safetensors",
                output_dir=str(Path(tmp) / "out"),
                output_name="comfy_inputs_case",
                backend_workdir=str(REPO_ROOT),
                python_executable=sys.executable,
                entrypoint_mode="module",
                module_name="tests.fake_onetrainer_backend",
                script_path="",
                network_dim=16,
                network_alpha=16,
                learning_rate=1e-4,
                max_train_steps=100,
                lr_scheduler="cosine",
                resolution_x=512,
                resolution_y=512,
                batch_size_override=0,
                gradient_accumulation_override=0,
                execute=False,
                extra_args="",
                runtime_root=str(Path(tmp) / "runtime"),
                model=object(),
                positive=[["cond", {"pooled_output": None}]],
                latents={"samples": FakeTensor((1, 4, 64, 64))},
            )
        self.assertTrue(job["config_payload"]["preview"]["uses_comfy_preview_inputs"])
        self.assertTrue(job["config_payload"]["preview"]["connected_inputs"]["positive_conditioning"]["connected"])

    def test_train_lora_node_returns_lora_and_loss_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            node = self.TrainLoRANode()
            lora, loss_map, steps = node.run(
                model=object(),
                latents={"samples": FakeTensor((1, 4, 64, 64))},
                positive=[["cond", {"pooled_output": None}]],
                dataset_dir=self._make_dataset_dir(tmp),
                model_family="flux",
                batch_size=1,
                grad_accumulation_steps=1,
                steps=16,
                learning_rate=0.0005,
                rank=8,
                resolution_x=1024,
                resolution_y=1024,
                optimizer="AdamW",
                loss_function="MSE",
                seed=0,
                control_after_generate="randomize",
                training_dtype="bf16",
                lora_dtype="bf16",
                quantized_backward=False,
                algorithm="LoRA",
                gradient_checkpointing=True,
                checkpoint_depth=1,
                offloading=False,
                existing_lora="[None]",
                bucket_mode=False,
                bypass_mode=False,
                sample_every_n_steps=10,
                sample_prompts="portrait photo",
                sampler="euler_a",
                sample_steps=20,
                cfg_scale=7.0,
            )
        self.assertEqual(steps, 16)
        self.assertEqual(lora["steps"], 16)
        self.assertEqual(loss_map["status"], "dry_run")
        self.assertEqual(lora["training_dtype"], "bf16")

    def test_train_lora_advanced_returns_lora_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            validated_dataset = self._make_validated_dataset(tmp)
            precision_plan = self._make_precision()
            preview_config = self.build_preview_config(
                enabled=True,
                sample_every_n_steps=20,
                sample_prompts="portrait photo",
                sampler="euler_a",
                sample_steps=20,
                cfg_scale=7.0,
                seed=42,
            )
            node = self.TrainLoRAAdvancedNode()
            lora, preview_image, loss_map, steps = node.run(
                validated_dataset=validated_dataset,
                precision_plan=precision_plan,
                preview_config=preview_config,
                rank=16,
                learning_rate=0.0001,
                max_train_steps=120,
                lr_scheduler="cosine",
                resolution_x=768,
                resolution_y=1024,
                batch_size_override=1,
                gradient_accumulation_override=2,
                optimizer="AdamW8bit",
                loss_function="Huber",
                seed=123,
                control_after_generate="fixed",
                training_dtype="fp8",
                lora_dtype="bf16",
                quantized_backward=False,
                algorithm="LoRA",
                gradient_checkpointing=True,
                checkpoint_depth=1,
                offloading=False,
                existing_lora="[None]",
                bucket_mode=False,
                bypass_mode=False,
                model=object(),
                positive=[["cond", {"pooled_output": None}]],
                latents={"samples": FakeTensor((1, 4, 64, 64))},
            )
        self.assertEqual(steps, 120)
        self.assertEqual(lora["steps"], 120)
        self.assertEqual(lora["training_dtype"], "fp8")
        self.assertEqual(lora["lora_dtype"], "bf16")
        self.assertEqual(loss_map["status"], "dry_run")
        self.assertIsNone(preview_image)

    def test_background_job_and_status_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._make_validated_dataset(tmp)
            precision = self._make_precision()
            preview = self.build_preview_config(
                enabled=True,
                sample_every_n_steps=10,
                sample_prompts="portrait photo",
                sampler="euler_a",
                sample_steps=20,
                cfg_scale=7.0,
                seed=42,
            )
            runtime_root = str(Path(tmp) / "runtime")
            job = self.prepare_training_run(
                validated_dataset=dataset,
                precision_plan=precision,
                preview_config=preview,
                base_model_path="/models/base.safetensors",
                output_dir=str(Path(tmp) / "out"),
                output_name="success_case",
                backend_workdir=str(REPO_ROOT),
                python_executable=sys.executable,
                entrypoint_mode="module",
                module_name="tests.fake_onetrainer_backend",
                script_path="",
                network_dim=16,
                network_alpha=16,
                learning_rate=1e-4,
                max_train_steps=100,
                lr_scheduler="cosine",
                resolution_x=512,
                resolution_y=512,
                batch_size_override=0,
                gradient_accumulation_override=0,
                execute=True,
                extra_args="",
                runtime_root=runtime_root,
            )
            resolved = self._wait_for_job(job, runtime_root)
        self.assertEqual(resolved["status"], "completed")
        self.assertEqual(resolved["returncode"], 0)
        self.assertTrue(resolved["latest_preview"].endswith(".png"))
        self.assertAlmostEqual(resolved["latest_loss"], 0.0456, places=4)
        self.assertIn("loss=0.0456", resolved["latest_loss_line"])
        self.assertIn("Training completed", resolved["log_tail"])

    def test_save_lora_weights_node_copies_generated_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            loras_dir = Path(tmp) / "loras"
            os.environ["DYNATRAIN_LORAS_DIR"] = str(loras_dir)
            try:
                dataset = self._make_validated_dataset(tmp)
                precision = self._make_precision()
                preview = self.build_preview_config(
                    enabled=True,
                    sample_every_n_steps=10,
                    sample_prompts="portrait photo",
                    sampler="euler_a",
                    sample_steps=20,
                    cfg_scale=7.0,
                    seed=42,
                )
                job = self.prepare_training_run(
                    validated_dataset=dataset,
                    precision_plan=precision,
                    preview_config=preview,
                    base_model_path="/models/base.safetensors",
                    output_dir=str(Path(tmp) / "out"),
                    output_name="save_case",
                    backend_workdir=str(REPO_ROOT),
                    python_executable=sys.executable,
                    entrypoint_mode="module",
                    module_name="tests.fake_onetrainer_backend",
                    script_path="",
                    network_dim=16,
                    network_alpha=16,
                    learning_rate=1e-4,
                    max_train_steps=33,
                    lr_scheduler="cosine",
                    resolution_x=512,
                    resolution_y=512,
                    batch_size_override=0,
                    gradient_accumulation_override=0,
                    execute=True,
                    extra_args="",
                    runtime_root=str(Path(tmp) / "runtime"),
                )
                resolved = self._wait_for_job(job, str(Path(tmp) / "runtime"))
                lora_payload = self.build_lora_payload(resolved)
                save_node = self.SaveLoRAWeightsNode()
                saved_lora, saved_path, _summary = save_node.run(lora_payload, "loras/test-prefix", 33)
                self.assertTrue(Path(saved_path).exists())
                self.assertTrue(saved_path.endswith("test-prefix_33.safetensors"))
                self.assertEqual(saved_lora["weights_path"], saved_path)
            finally:
                os.environ.pop("DYNATRAIN_LORAS_DIR", None)

    def test_plot_loss_graph_node_saves_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._make_validated_dataset(tmp)
            precision = self._make_precision()
            preview = self.build_preview_config(
                enabled=True,
                sample_every_n_steps=10,
                sample_prompts="portrait photo",
                sampler="euler_a",
                sample_steps=20,
                cfg_scale=7.0,
                seed=42,
            )
            runtime_root = str(Path(tmp) / "runtime")
            job = self.prepare_training_run(
                validated_dataset=dataset,
                precision_plan=precision,
                preview_config=preview,
                base_model_path="/models/base.safetensors",
                output_dir=str(Path(tmp) / "out"),
                output_name="graph_case",
                backend_workdir=str(REPO_ROOT),
                python_executable=sys.executable,
                entrypoint_mode="module",
                module_name="tests.fake_onetrainer_backend",
                script_path="",
                network_dim=16,
                network_alpha=16,
                learning_rate=1e-4,
                max_train_steps=33,
                lr_scheduler="cosine",
                resolution_x=512,
                resolution_y=512,
                batch_size_override=0,
                gradient_accumulation_override=0,
                execute=True,
                extra_args="",
                runtime_root=runtime_root,
            )
            resolved = self._wait_for_job(job, runtime_root)
            loss_map = self.build_loss_map(resolved)
            history = self.loss_history(loss_map["log_path"])
            self.assertEqual(len(history), 3)
            plot_node = self.PlotLossGraphNode()
            graph_image, saved_path, summary = plot_node.run(loss_map, "loss_graph/test")
        self.assertIsNone(graph_image)
        self.assertTrue(Path(saved_path).exists())
        self.assertTrue(saved_path.endswith(".png"))
        self.assertIn("\"points\": 3", summary)

    def test_background_job_and_status_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._make_validated_dataset(tmp)
            precision = self._make_precision()
            preview = self.build_preview_config(
                enabled=True,
                sample_every_n_steps=10,
                sample_prompts="portrait photo",
                sampler="euler_a",
                sample_steps=20,
                cfg_scale=7.0,
                seed=42,
            )
            runtime_root = str(Path(tmp) / "runtime")
            job = self.prepare_training_run(
                validated_dataset=dataset,
                precision_plan=precision,
                preview_config=preview,
                base_model_path="/models/base.safetensors",
                output_dir=str(Path(tmp) / "out"),
                output_name="failure_case",
                backend_workdir=str(REPO_ROOT),
                python_executable=sys.executable,
                entrypoint_mode="script",
                module_name="",
                script_path=str(REPO_ROOT / "tests" / "fake_onetrainer_backend.py"),
                network_dim=16,
                network_alpha=16,
                learning_rate=1e-4,
                max_train_steps=100,
                lr_scheduler="cosine",
                resolution_x=512,
                resolution_y=512,
                batch_size_override=0,
                gradient_accumulation_override=0,
                execute=True,
                extra_args="--fail",
                runtime_root=runtime_root,
            )
            resolved = self._wait_for_job(job, runtime_root)
        self.assertEqual(resolved["status"], "failed")
        self.assertEqual(resolved["returncode"], 3)
        self.assertAlmostEqual(resolved["latest_loss"], 0.9876, places=4)
        self.assertIn("loss=0.9876", resolved["latest_loss_line"])
        self.assertIn("Intentional backend failure", resolved["log_tail"])

    def test_missing_module_gives_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._make_validated_dataset(tmp)
            precision = self._make_precision()
            preview = self.build_preview_config(
                enabled=True,
                sample_every_n_steps=10,
                sample_prompts="portrait photo",
                sampler="euler_a",
                sample_steps=20,
                cfg_scale=7.0,
                seed=42,
            )
            with self.assertRaises(FileNotFoundError) as ctx:
                self.prepare_training_run(
                    validated_dataset=dataset,
                    precision_plan=precision,
                    preview_config=preview,
                    base_model_path="/models/base.safetensors",
                    output_dir=str(Path(tmp) / "out"),
                    output_name="missing_module_case",
                    backend_workdir=str(REPO_ROOT),
                    python_executable=sys.executable,
                    entrypoint_mode="module",
                    module_name="definitely_missing_onetrainer_module_12345",
                    script_path="",
                    network_dim=16,
                    network_alpha=16,
                    learning_rate=1e-4,
                    max_train_steps=100,
                    lr_scheduler="cosine",
                    resolution_x=512,
                    resolution_y=512,
                    batch_size_override=0,
                    gradient_accumulation_override=0,
                    execute=True,
                    extra_args="",
                    runtime_root=str(Path(tmp) / "runtime"),
                )
        self.assertIn("Install OneTrainer", str(ctx.exception))

    def test_dry_run_reports_backend_not_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._make_validated_dataset(tmp)
            precision = self._make_precision()
            preview = self.build_preview_config(
                enabled=True,
                sample_every_n_steps=10,
                sample_prompts="portrait photo",
                sampler="euler_a",
                sample_steps=20,
                cfg_scale=7.0,
                seed=42,
            )
            job = self.prepare_training_run(
                validated_dataset=dataset,
                precision_plan=precision,
                preview_config=preview,
                base_model_path="/models/base.safetensors",
                output_dir=str(Path(tmp) / "out"),
                output_name="dry_missing_module_case",
                backend_workdir=str(REPO_ROOT),
                python_executable=sys.executable,
                entrypoint_mode="module",
                module_name="definitely_missing_onetrainer_module_12345",
                script_path="",
                network_dim=16,
                network_alpha=16,
                learning_rate=1e-4,
                max_train_steps=100,
                lr_scheduler="cosine",
                resolution_x=512,
                resolution_y=512,
                batch_size_override=0,
                gradient_accumulation_override=0,
                execute=False,
                extra_args="",
                runtime_root=str(Path(tmp) / "runtime"),
            )
        self.assertFalse(job["backend_ready"])
        self.assertIn("Install OneTrainer", job["backend_check_message"])

    def test_missing_job_status_raises_clean_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                self.get_job_status(job_id="missing-job", runtime_root=str(Path(tmp) / "runtime"))

    def _make_validated_dataset(self, tmp: str):
        root = Path(self._make_dataset_dir(tmp))
        report = self.summarize_dataset(str(root))
        return self.validate_caption_pairs(dataset_report=report)

    def _make_dataset_dir(self, tmp: str):
        root = Path(tmp) / "dataset"
        root.mkdir(parents=True, exist_ok=True)
        (root / "a.jpg").write_bytes(b"image-a")
        (root / "b.jpg").write_bytes(b"image-b")
        (root / "a.txt").write_text("portrait photo", encoding="utf-8")
        (root / "b.txt").write_text("editorial portrait", encoding="utf-8")
        return str(root)

    def _make_precision(self):
        return self.plan_precision(
            model_family="sd15",
            vram_gb=12.0,
            requested_precision="auto",
            save_precision="auto",
            optimizer="adamw8bit",
            gradient_accumulation_steps=2,
            gpu_supports_bf16=True,
            allow_experimental_fp8=False,
        )

    def _wait_for_job(self, job, runtime_root: str):
        resolved = job
        for _ in range(80):
            resolved = self.get_job_status(train_job=job, runtime_root=runtime_root)
            if resolved["status"] in {"completed", "failed"}:
                return resolved
            time.sleep(0.1)
        self.fail(f"Job did not finish in time: {json.dumps(resolved, indent=2)}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
