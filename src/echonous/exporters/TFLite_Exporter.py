"""
TFLite export API for converting a PyTorch nn.Module (via Model dataclass) to TFLite.

Requirements/assumptions:
- Python 3.8+
- Prefer TensorFlow 2.x runtime
- Default path: PyTorch -> ONNX -> TF SavedModel -> TFLite (CPU for validation)
- Optional NoBuCo path: PyTorch -> TF/Keras -> TFLite (custom op replacement via @nobuco.converter)
- Supports FP32 and FP16. INT8 not in scope.
- Inputs are specified by `Model` dataclass; TFLite prefers NHWC.
- Validation compares PyTorch vs TFLite outputs with absolute tolerance.

Notes:
- GPU preference: prefer TensorFlow conversion pipeline where feasible. Validation remains CPU for reproducibility.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Tuple
import os
import json
import time
import tempfile

import numpy as np
import torch
import torch.nn as nn

from model_class import Model, ModuleInputOutput


class TFLiteExportError(Exception):
    pass


class TFLiteExportAPI:
    def __init__(self) -> None:
        self.custom_op_handlers: Dict[str, Callable[[str], Optional[nn.Module]]] = {}

    def handleCustomOp(self, name: str, handler: Callable[[str], Optional[nn.Module]]) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("Custom op name must be a non-empty string")
        if not callable(handler):
            raise ValueError("Custom op handler must be callable")
        self.custom_op_handlers[name] = handler

    def convert_model_dataclass(
        self,
        model_dc: Model,
        output_path: str,
        precision: str = "Float32",  # 'Float32' | 'Float16'
        use_nobuco: bool = False,
        tolerance: float = 1e-3,
        flexible_shapes: bool = False,
        debug: bool = False,
    ) -> Dict[str, Any]:
        start_time = time.time()

        if flexible_shapes:
            self._handle_flexible_shapes_stub()

        module = model_dc.module
        if not isinstance(module, nn.Module):
            raise TFLiteExportError("model_dc.module must be an instance of torch.nn.Module")

        module.eval()

        torch_inputs = self._build_torch_inputs(model_dc.inputs)

        if use_nobuco:
            tflite_bytes, tf_version, tflite_version = self._convert_with_nobuco(module, model_dc, torch_inputs, precision)
        else:
            # If custom ops registered but not using NoBuCo, instruct user to enable NoBuCo
            if self.custom_op_handlers:
                raise TFLiteExportError(
                    "Custom ops require use_nobuco=True with @nobuco.converter handlers."
                )
            tflite_bytes, tf_version, tflite_version = self._convert_via_onnx(module, model_dc, torch_inputs, precision)

        # Attempt flatbuffer metadata embedding (best-effort)
        try:
            tflite_bytes = self._embed_flatbuffer_metadata(
                tflite_bytes=tflite_bytes,
                model_dc=model_dc,
                precision=precision,
                use_nobuco=use_nobuco,
                tf_version=tf_version,
                tflite_version=tflite_version,
            )
        except Exception:
            pass

        # Save model
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(tflite_bytes)

        # Validate on CPU
        validation, debug_info = self._validate_tflite(module, model_dc, torch_inputs, tflite_bytes, tolerance, debug)

        # Sidecar metadata for now (FlatBuffer metadata embedding is optional and TBD)
        sidecar_path = self._write_sidecar_metadata(
            output_path, model_dc, precision, use_nobuco, tf_version, tflite_version, tolerance, validation
        )

        # Debug JSON
        debug_path = None
        if debug and debug_info is not None:
            debug_path = output_path.replace('.tflite', '_debug.json') if output_path.endswith('.tflite') else f"{output_path}_debug.json"
            try:
                with open(debug_path, 'w') as f:
                    json.dump(debug_info, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else o)
            except Exception:
                debug_path = None

        return {
            "output_model_path": output_path,
            "sidecar_metadata_path": sidecar_path,
            "tensorflow_version": tf_version,
            "tflite_version": tflite_version,
            "precision": precision,
            "use_nobuco": use_nobuco,
            "validation": validation,
            "elapsed_seconds": round(time.time() - start_time, 3),
            "debug_path": debug_path,
        }

    def _build_torch_inputs(self, inputs: List[ModuleInputOutput]) -> List[torch.Tensor]:
        tensors: List[torch.Tensor] = []
        for io in inputs:
            shape = tuple(int(s) for s in io.shape)
            if any(dim <= 0 for dim in shape):
                raise TFLiteExportError(f"Input '{io.name}' has non-positive dimension in shape: {shape}")
            tensors.append(torch.rand(shape, dtype=torch.float32))
        return tensors

    def _convert_via_onnx(
        self,
        module: nn.Module,
        model_dc: Model,
        example_inputs: List[torch.Tensor],
        precision: str,
    ) -> Tuple[bytes, str, str]:
        try:
            import onnx  # type: ignore
            import tensorflow as tf  # type: ignore
        except Exception as e:
            raise TFLiteExportError(f"Required packages missing for ONNX path: {e}")

        # Export PyTorch -> ONNX
        onnx_path = os.path.join("/tmp", f"_export_{int(time.time()*1e6)}.onnx")
        self._export_pytorch_to_onnx(module, model_dc, example_inputs, onnx_path)

        # Convert ONNX -> TFLite (prefer going through TF SavedModel if needed)
        # Simplify by loading ONNX with onnx and using TF importer (onnx-tf) if available; else try direct tf-onnx
        try:
            import onnx_tf  # type: ignore
            from onnx_tf.backend import prepare  # type: ignore
            onnx_model = onnx.load(onnx_path)
            tf_rep = prepare(onnx_model)
            saved_model_dir = os.path.join("/tmp", f"_saved_{int(time.time()*1e6)}")
            tf_rep.export_graph(saved_model_dir)
            converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        except Exception:
            # Fallback: attempt tf-onnx conversion to GraphDef/Keras if available, otherwise raise
            try:
                import tf2onnx  # type: ignore  # noqa: F401
                # Without a robust tf2onnx->tf graph path, raise a clearer error
                raise TFLiteExportError(
                    "onnx-tf is required for ONNX->TF conversion in this pipeline. Install 'onnx-tf'."
                )
            finally:
                pass

        # Precision
        if precision == "Float16":
            try:
                converter.target_spec.supported_types = [tf.float16]
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
            except Exception:
                pass

        try:
            tflite_bytes = converter.convert()
        except Exception as e:
            raise TFLiteExportError(f"TFLite conversion failed: {e}")

        tf_version = getattr(tf, "__version__", "unknown")
        try:
            import tflite_runtime as _rt  # type: ignore
            tflite_version = getattr(_rt, "__version__", "runtime")
        except Exception:
            tflite_version = tf_version  # approximate

        return tflite_bytes, tf_version, tflite_version

    def _export_pytorch_to_onnx(
        self,
        module: nn.Module,
        model_dc: Model,
        example_inputs: List[torch.Tensor],
        onnx_path: str,
    ) -> None:
        module_cpu = module.cpu().eval()
        # Build dummy inputs; transpose to NHWC for exporter graph alignment when practical
        torch_inputs = []
        for idx, io in enumerate(model_dc.inputs):
            x = example_inputs[idx].detach().cpu()
            # ONNX export often expects NCHW from PyTorch; keep NCHW for tracing
            torch_inputs.append(x)

        dynamic_axes = {io.name: {0: 'batch'} for io in model_dc.inputs}
        input_names = [io.name for io in model_dc.inputs]
        output_names = [io.name for io in model_dc.outputs] if model_dc.outputs else None

        try:
            torch.onnx.export(
                module_cpu,
                tuple(torch_inputs) if len(torch_inputs) > 1 else torch_inputs[0],
                onnx_path,
                input_names=input_names,
                output_names=output_names,
                opset_version=13,
                do_constant_folding=True,
                dynamic_axes=dynamic_axes,
            )
        except Exception as e:
            raise TFLiteExportError(f"PyTorch->ONNX export failed: {e}")

    def _convert_with_nobuco(
        self,
        module: nn.Module,
        model_dc: Model,
        example_inputs: List[torch.Tensor],
        precision: str,
    ) -> Tuple[bytes, str, str]:
        try:
            import tensorflow as tf  # type: ignore
            import nobuco  # type: ignore
        except Exception as e:
            raise TFLiteExportError(f"NoBuCo path requires tensorflow and nobuco installed: {e}")

        # NoBuCo expects annotated converters via @nobuco.converter within user code
        # Here we simply call nobuco.convert to get a TF function/Keras model
        try:
            tf_model = nobuco.convert(module, example_inputs=tuple(example_inputs) if len(example_inputs) > 1 else example_inputs[0])
        except Exception as e:
            raise TFLiteExportError(f"NoBuCo conversion failed: {e}")

        # Export SavedModel
        saved_model_dir = os.path.join("/tmp", f"_saved_{int(time.time()*1e6)}")
        try:
            tf.saved_model.save(tf_model, saved_model_dir)  # type: ignore[arg-type]
        except Exception as e:
            raise TFLiteExportError(f"Failed to save TF model: {e}")

        # TFLite conversion
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        if precision == "Float16":
            try:
                converter.target_spec.supported_types = [tf.float16]
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
            except Exception:
                pass
        try:
            tflite_bytes = converter.convert()
        except Exception as e:
            raise TFLiteExportError(f"TFLite conversion (NoBuCo) failed: {e}")

        tf_version = getattr(tf, "__version__", "unknown")
        try:
            import tflite_runtime as _rt  # type: ignore
            tflite_version = getattr(_rt, "__version__", "runtime")
        except Exception:
            tflite_version = tf_version

        return tflite_bytes, tf_version, tflite_version

    def _validate_tflite(
        self,
        module: nn.Module,
        model_dc: Model,
        torch_inputs: List[torch.Tensor],
        tflite_bytes: bytes,
        tolerance: float,
        debug: bool,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        # PyTorch forward (NCHW)
        module_cpu = module.cpu().eval()
        with torch.no_grad():
            pt_out = module_cpu(*torch_inputs) if len(torch_inputs) > 1 else module_cpu(torch_inputs[0])

        pt_outputs: List[np.ndarray] = []
        if isinstance(pt_out, (list, tuple)):
            for o in pt_out:
                pt_outputs.append(self._to_numpy(o))
        else:
            pt_outputs.append(self._to_numpy(pt_out))

        # TFLite inference (NHWC)
        try:
            import tensorflow as tf  # type: ignore
        except Exception as e:
            return {"passed": False, "error": f"TensorFlow not available for validation: {e}"}, None

        interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Build inputs matching interpreter's expected layout per tensor
        feed_summaries: List[Dict[str, Any]] = []
        for idx, io in enumerate(model_dc.inputs):
            x_nchw = torch_inputs[idx].detach().cpu().numpy().astype(np.float32)
            x_nhwc = np.transpose(x_nchw, (0, 2, 3, 1)) if x_nchw.ndim == 4 else x_nchw

            # Apply image scale regardless of layout
            if io.type == 'image':
                try:
                    s = float(io.scale)
                    x_nchw = x_nchw * s
                    x_nhwc = x_nhwc * s
                except Exception:
                    pass

            expected = tuple(int(d) for d in input_details[idx]['shape'])
            expected_dtype = input_details[idx]['dtype']

            def matches_shape(arr_shape: Tuple[int, ...], exp: Tuple[int, ...]) -> bool:
                if len(arr_shape) != len(exp):
                    return False
                for a, e in zip(arr_shape, exp):
                    if e == -1:
                        continue
                    if a != e:
                        return False
                return True

            # Prefer NHWC if it matches expected; otherwise use NCHW if that matches
            if matches_shape(x_nhwc.shape, expected):
                x_feed = x_nhwc
            elif matches_shape(x_nchw.shape, expected):
                x_feed = x_nchw
            else:
                # If neither matches, attempt best-effort: if expected looks NHWC
                if len(expected) == 4 and expected[-1] in (1, 3):
                    x_feed = x_nhwc
                else:
                    x_feed = x_nchw

            # Cast dtype to what interpreter expects
            if x_feed.dtype != expected_dtype:
                try:
                    x_feed = x_feed.astype(expected_dtype)
                except Exception:
                    pass

            # Set to interpreter
            interpreter.set_tensor(input_details[idx]['index'], x_feed)

            if debug:
                def stats(arr: np.ndarray) -> Dict[str, Any]:
                    return {
                        "shape": list(arr.shape),
                        "dtype": str(arr.dtype),
                        "min": float(np.min(arr)) if arr.size else 0.0,
                        "max": float(np.max(arr)) if arr.size else 0.0,
                        "mean": float(np.mean(arr)) if arr.size else 0.0,
                        "std": float(np.std(arr)) if arr.size else 0.0,
                    }
                feed_summaries.append({
                    "name": io.name,
                    "expected_shape": list(expected),
                    "expected_dtype": str(expected_dtype),
                    "x_nchw": stats(x_nchw),
                    "x_nhwc": stats(x_nhwc),
                    "x_feed": stats(x_feed),
                    "scale": float(io.scale) if io.type == 'image' else None,
                })

        try:
            interpreter.invoke()
        except Exception as e:
            return {"passed": False, "error": f"TFLite inference failed: {e}"}, None

        tfl_outs: List[np.ndarray] = []
        for od in output_details:
            tfl_outs.append(interpreter.get_tensor(od['index']))

        # Compare outputs (squeeze NHWC dims to match PyTorch shapes where applicable)
        comparisons: List[Dict[str, Any]] = []
        passed = True
        for i in range(min(len(pt_outputs), len(tfl_outs))):
            a = pt_outputs[i]
            b = tfl_outs[i]
            # Flatten batch and trailing singleton dims
            a_cmp = a
            b_cmp = b
            while a_cmp.ndim > 1 and a_cmp.shape[0] == 1:
                a_cmp = a_cmp.squeeze(0)
            while b_cmp.ndim > 1 and b_cmp.shape[0] == 1:
                b_cmp = b_cmp.squeeze(0)
            same_shape = a_cmp.shape == b_cmp.shape
            if not same_shape:
                passed = False
                diff = float("inf")
            else:
                diff = float(np.max(np.abs(a_cmp - b_cmp))) if a_cmp.size and b_cmp.size else 0.0
                if not np.allclose(a_cmp, b_cmp, atol=tolerance, rtol=0):
                    passed = False
            comparisons.append({
                "index": i,
                "pytorch_shape": list(a.shape),
                "tflite_shape": list(b.shape),
                "max_abs_diff": None if not same_shape else round(diff, 6),
                "within_tolerance": same_shape and (diff <= tolerance),
            })

        debug_info: Optional[Dict[str, Any]] = None
        if debug:
            # Optional ONNXRuntime check
            ort_summary = None
            try:
                import onnxruntime as ort  # type: ignore
                # Build ORT session from ONNX export to compare outputs
                onnx_tmp = os.path.join("/tmp", f"_debug_{int(time.time()*1e6)}.onnx")
                self._export_pytorch_to_onnx(module, model_dc, torch_inputs, onnx_tmp)
                sess = ort.InferenceSession(onnx_tmp, providers=["CPUExecutionProvider"])  # type: ignore
                ort_inputs = {}
                for i, io in enumerate(model_dc.inputs):
                    ort_inputs[sess.get_inputs()[i].name] = torch_inputs[i].detach().cpu().numpy().astype(np.float32)
                ort_outs = sess.run(None, ort_inputs)
                ort_diffs = []
                for i, a in enumerate(pt_outputs):
                    b = ort_outs[i]
                    diff = float(np.max(np.abs(a - b))) if a.size and np.array(b).size else 0.0
                    ort_diffs.append({"index": i, "max_abs_diff": diff})
                ort_summary = {"diffs": ort_diffs}
            except Exception as e:
                ort_summary = {"error": str(e)}

            debug_info = {
                "inputs": feed_summaries,
                "onnxruntime": ort_summary,
            }

        return {"passed": passed, "tolerance": tolerance, "comparisons": comparisons}, debug_info

    def _to_numpy(self, x: Any) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        if isinstance(x, np.ndarray):
            return x
        return np.array(x)

    def _write_sidecar_metadata(
        self,
        output_path: str,
        model_dc: Model,
        precision: str,
        use_nobuco: bool,
        tf_version: str,
        tflite_version: str,
        tolerance: float,
        validation: Dict[str, Any],
    ) -> str:
        meta = {
            "name": model_dc.name,
            "version": model_dc.version,
            "description": model_dc.description,
            "inputs": [asdict(i) for i in model_dc.inputs],
            "outputs": [asdict(o) for o in model_dc.outputs],
            "precision": precision,
            "use_nobuco": use_nobuco,
            "tensorflow_version": tf_version,
            "tflite_version": tflite_version,
            "tolerance": tolerance,
            "validation": validation,
            "generated_at": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        }
        sidecar = output_path.replace('.tflite', '_metadata.json') if output_path.endswith('.tflite') else f"{output_path}_metadata.json"
        try:
            with open(sidecar, 'w') as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            raise TFLiteExportError(f"Failed to write sidecar metadata: {e}")
        return sidecar

    def _handle_flexible_shapes_stub(self) -> None:
        raise NotImplementedError(
            "Flexible input shape ranges are not implemented yet. "
            "Stub present for future implementation."
        )

    def _embed_flatbuffer_metadata(
        self,
        tflite_bytes: bytes,
        model_dc: Model,
        precision: str,
        use_nobuco: bool,
        tf_version: str,
        tflite_version: str,
    ) -> bytes:
        """
        Embed minimal ModelMetadata and I/O TensorMetadata into the TFLite flatbuffer.
        Returns updated TFLite bytes. Raises TFLiteExportError if tooling is missing.
        """
        try:
            from tflite_support.metadata import metadata as _md  # type: ignore
            from tflite_support.metadata import schema_py_generated as _mfb  # type: ignore
            import flatbuffers  # type: ignore
        except Exception as e:
            raise TFLiteExportError(f"tflite-support not available for metadata embedding: {e}")

        builder = flatbuffers.Builder(2048)

        name_off = builder.CreateString(str(model_dc.name))
        desc_off = builder.CreateString(str(model_dc.description or ""))
        ver_off = builder.CreateString(str(model_dc.version))

        # Input TensorMetadata
        input_tm_offsets: List[int] = []
        for io in model_dc.inputs:
            io_name = builder.CreateString(io.name)
            _mfb.TensorMetadataStart(builder)
            _mfb.TensorMetadataAddName(builder, io_name)
            tm = _mfb.TensorMetadataEnd(builder)
            input_tm_offsets.append(tm)

        # Output TensorMetadata
        output_tm_offsets: List[int] = []
        for oo in model_dc.outputs:
            oo_name = builder.CreateString(oo.name)
            _mfb.TensorMetadataStart(builder)
            _mfb.TensorMetadataAddName(builder, oo_name)
            tm = _mfb.TensorMetadataEnd(builder)
            output_tm_offsets.append(tm)

        # SubGraphMetadata vectors
        _mfb.SubGraphMetadataStartInputTensorMetadataVector(builder, len(input_tm_offsets))
        for tm in reversed(input_tm_offsets):
            builder.PrependUOffsetTRelative(tm)
        in_vec = builder.EndVector(len(input_tm_offsets))

        _mfb.SubGraphMetadataStartOutputTensorMetadataVector(builder, len(output_tm_offsets))
        for tm in reversed(output_tm_offsets):
            builder.PrependUOffsetTRelative(tm)
        out_vec = builder.EndVector(len(output_tm_offsets))

        _mfb.SubGraphMetadataStart(builder)
        _mfb.SubGraphMetadataAddName(builder, name_off)
        _mfb.SubGraphMetadataAddInputTensorMetadata(builder, in_vec)
        _mfb.SubGraphMetadataAddOutputTensorMetadata(builder, out_vec)
        sgm = _mfb.SubGraphMetadataEnd(builder)

        # ModelMetadata
        _mfb.ModelMetadataStartSubgraphMetadataVector(builder, 1)
        builder.PrependUOffsetTRelative(sgm)
        sgv = builder.EndVector(1)

        _mfb.ModelMetadataStart(builder)
        _mfb.ModelMetadataAddName(builder, name_off)
        _mfb.ModelMetadataAddDescription(builder, desc_off)
        _mfb.ModelMetadataAddVersion(builder, ver_off)
        _mfb.ModelMetadataAddSubgraphMetadata(builder, sgv)
        mm = _mfb.ModelMetadataEnd(builder)
        builder.Finish(mm)
        metadata_buf = bytes(builder.Output())

        # Populate metadata into the flatbuffer using a temp file
        tmp = tempfile.NamedTemporaryFile(suffix='.tflite', delete=False)
        tmp_name = tmp.name
        try:
            tmp.write(tflite_bytes)
            tmp.flush()
            tmp.close()

            pop = _md.MetadataPopulator.with_model_file(tmp_name)
            pop.load_metadata_buffer(metadata_buf)
            pop.populate()

            with open(tmp_name, 'rb') as f:
                return f.read()
        finally:
            try:
                os.unlink(tmp_name)
            except Exception:
                pass


def convert_model_to_tflite(
    model_dc: Model,
    output_path: str,
    precision: str = "Float32",
    use_nobuco: bool = False,
    tolerance: float = 1e-3,
    flexible_shapes: bool = False,
) -> Dict[str, Any]:
    api = TFLiteExportAPI()
    return api.convert_model_dataclass(
        model_dc,
        output_path=output_path,
        precision=precision,
        use_nobuco=use_nobuco,
        tolerance=tolerance,
        flexible_shapes=flexible_shapes,
        debug=True
    )


