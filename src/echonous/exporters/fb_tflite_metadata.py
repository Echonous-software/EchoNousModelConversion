from __future__ import annotations

import traceback
import sys
import os
from typing import List


def build_model_metadata_buffer(
    name: str,
    description: str | None,
    version: str | int | None,
    input_names: List[str],
    output_names: List[str],
) -> bytes:
    """
    Build a minimal TFLite ModelMetadata flatbuffer using FlatBuffers only.

    Fields populated:
      - ModelMetadata.name, description, version
      - SubGraphMetadata[0].name
      - SubGraphMetadata[0].input_tensor_metadata[*].name
      - SubGraphMetadata[0].output_tensor_metadata[*].name
    """
    print("###### BUILDING MODEL METADATA BUFFER ######")
    # Flatbuffer schema validation debugging
    print(f"Schema file 'tflite.fbs' exists: {os.path.exists('tflite.fbs')}")
    print(f"Schema file 'tflite_metadata.fbs' exists: {os.path.exists('tflite_metadata.fbs')}")
    print(f"Generated tflite modules available: {'tflite' in sys.modules}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Files in current directory: {os.listdir('.')}")
    
    try:
        import flatbuffers  # type: ignore
        print(f"Flatbuffers version: {getattr(flatbuffers, '__version__', 'unknown')}")
    except Exception as e:
        print(f"Flatbuffers import failed: {e}")
        raise RuntimeError("flatbuffers package is required to build metadata buffer") from e

    builder = flatbuffers.Builder(2048)

    name_off = builder.CreateString(str(name))
    desc_off = builder.CreateString(str(description or ""))
    ver_off = builder.CreateString(str(version)) if version is not None else builder.CreateString("")

    # Build TensorMetadata entries (name only)
    def build_tensor_metadata(nm: str) -> int:
        print(f"Building tensor metadata for {nm}")
        n_off = builder.CreateString(nm)
        builder.StartObject(1)
        builder.PrependUOffsetTRelativeSlot(0, n_off, 0)
        return builder.EndObject()
    print(f"INPUT COUNT {len(input_names)}")
    print(f"OUTPUT COUNT {len(output_names)}")
    input_tm_offsets = [build_tensor_metadata(n) for n in input_names]
    output_tm_offsets = [build_tensor_metadata(n) for n in output_names]

    # Build vectors
    def build_vector(objs: List[int]) -> int:
        builder.StartVector(4, len(objs), 4)
        for off in reversed(objs):
            builder.PrependUOffsetTRelative(off)
        return builder.EndVector(len(objs))

    in_vec = build_vector(input_tm_offsets)
    out_vec = build_vector(output_tm_offsets)

    # SubGraphMetadata: fields: 0-name, 1-inputs, 2-outputs
    builder.StartObject(3)
    builder.PrependUOffsetTRelativeSlot(0, name_off, 0)
    builder.PrependUOffsetTRelativeSlot(1, in_vec, 0)
    builder.PrependUOffsetTRelativeSlot(2, out_vec, 0)
    sgm = builder.EndObject()

    # ModelMetadata: fields: 0-name,1-description,2-version,6-subgraph_metadata
    builder.StartVector(4, 1, 4)
    builder.PrependUOffsetTRelative(sgm)
    sgv = builder.EndVector(1)

    builder.StartObject(7)
    builder.PrependUOffsetTRelativeSlot(0, name_off, 0)
    builder.PrependUOffsetTRelativeSlot(1, desc_off, 0)
    builder.PrependUOffsetTRelativeSlot(2, ver_off, 0)
    builder.PrependUOffsetTRelativeSlot(6, sgv, 0)
    mm = builder.EndObject()
    # Add file identifier for tflite_metadata.fbs (M001)
    try:
        print("### BUILDER FINISH M001 ###")
        builder.Finish(mm, b"M001")
    except TypeError as e:
        # Older flatbuffers may not support identifier in Finish; fall back
        builder.Finish(mm)
        print("### EXCEPTION WHEN FINISHING METADATA ###")
        print(f"EXCEPTION TYPE: {type(e).__name__}")
        print(f"EXCEPTION MESSAGE: {str(e)}")
        print(f"EXCEPTION TRACEBACK: {traceback.format_exc()}")
    return bytes(builder.Output())


def embed_metadata_into_tflite_bytes(tflite_bytes: bytes, metadata_buf: bytes) -> bytes:
    """
    Pure-FlatBuffers embed using generated Python classes from the TFLite schema.
    Steps:
      1) Parse existing Model into object API (ModelT) if available.
      2) Append a new Buffer(Data=metadata_buf).
      3) Append a new Metadata(Buffer=<index>, Name="TFLite Metadata").
      4) Pack back to bytes with TFL3 identifier.

    Falls back to returning original bytes if object API not available.
    """
    try:
        print("### ATTEMPTING PURE FLATBUFFERS BUILD ###")
        
        # Schema validation debugging
        import tflite
        print(f"Schema files validation:")
        print(f"  tflite.fbs exists: {os.path.exists('tflite.fbs')}")
        print(f"  tflite_metadata.fbs exists: {os.path.exists('tflite_metadata.fbs')}")
        print(f"  Generated modules in sys.modules: {[k for k in sys.modules.keys() if 'tflite' in k]}")
        
        import flatbuffers  # type: ignore
        print(f"Flatbuffers imported successfully, version: {getattr(flatbuffers, '__version__', 'unknown')}")
        
        from echonous.exporters._tflite_schema.tflite_generated import Model, ModelT, BufferT, MetadataT
    except Exception as e:
        print("### Returning tflite_bytes early (no metadata) ###")
        print(f"EXCEPTION TYPE: {type(e).__name__}")
        print(f"EXCEPTION MESSAGE: {str(e)}")
        print(f"EXCEPTION TRACEBACK: {traceback.format_exc()}")
        return tflite_bytes

    # Require object API for safe rebuild
    if ModelT is None or BufferT is None or MetadataT is None:
        return tflite_bytes

    try:
        # Parse and convert to object API
        root = Model.GetRootAs(tflite_bytes, 0)
        
        # ---- Pre-change counts and buffer summary (original model) ----
        def _safe_call_len(obj, method_name: str) -> int:
            try:
                m = getattr(obj, method_name, None)
                if callable(m):
                    return int(m())
            except Exception:
                pass
            return 0
        
        def _collect_model_counts(tag: str, model_root) -> None:
            try:
                num_subgraphs = _safe_call_len(model_root, 'SubgraphsLength')
                num_buffers = _safe_call_len(model_root, 'BuffersLength')
                num_metadata = _safe_call_len(model_root, 'MetadataLength')
                # Sum per-subgraph operators and tensors
                total_operators = 0
                total_tensors = 0
                for si in range(num_subgraphs):
                    try:
                        sg = model_root.Subgraphs(si)
                        total_operators += _safe_call_len(sg, 'OperatorsLength')
                        total_tensors += _safe_call_len(sg, 'TensorsLength')
                    except Exception:
                        pass
                print(f"[{tag}] Counts -> subgraphs: {num_subgraphs}, operators: {total_operators}, tensors: {total_tensors}, buffers: {num_buffers}, metadata: {num_metadata}")
            except Exception as e:
                print(f"[{tag}] Failed to collect counts: {e}")
        
        def _get_buffer_sizes(model_root) -> List[int]:
            sizes: List[int] = []
            try:
                num_buffers = _safe_call_len(model_root, 'BuffersLength')
                for i in range(num_buffers):
                    try:
                        b = model_root.Buffers(i)
                        # Prefer DataLength() if available
                        if hasattr(b, 'DataLength') and callable(getattr(b, 'DataLength')):
                            sizes.append(int(b.DataLength()))
                        else:
                            # Fallbacks: attempt numpy/bytes helpers if present
                            if hasattr(b, 'DataAsNumpy') and callable(getattr(b, 'DataAsNumpy')):
                                try:
                                    arr = b.DataAsNumpy()
                                    sizes.append(int(len(arr)))
                                    continue
                                except Exception:
                                    pass
                            if hasattr(b, 'Data') and callable(getattr(b, 'Data')):
                                # Data(j) accessor exists, but no direct length; skip to 0 as last resort
                                sizes.append(0)
                            else:
                                sizes.append(0)
                    except Exception:
                        sizes.append(0)
            except Exception:
                pass
            return sizes
        
        def _print_buffer_summary(tag: str, model_root) -> List[int]:
            sizes = _get_buffer_sizes(model_root)
            if not sizes:
                print(f"[{tag}] Buffer summary unavailable or no buffers.")
                return sizes
            total = sum(sizes)
            nonzero = sum(1 for s in sizes if s > 0)
            print(f"[{tag}] Buffers -> count: {len(sizes)}, nonzero: {nonzero}, total_bytes: {total}")
            # Print first few entries as a sample
            sample_count = min(10, len(sizes))
            for i in range(sample_count):
                print(f"[{tag}]   Buffer {i}: {sizes[i]} bytes")
            if len(sizes) > sample_count:
                print(f"[{tag}]   ... {len(sizes) - sample_count} more buffers not shown")
            return sizes
        
        _collect_model_counts("ORIGINAL", root)
        orig_buf_sizes = _print_buffer_summary("ORIGINAL", root)
        print(f"Model class methods: {[m for m in dir(Model) if 'pack' in m.lower()]}")
        print(f"Model instance methods: {[m for m in dir(root) if 'pack' in m.lower()]}")
        print(f"ModelT class methods: {[m for m in dir(ModelT) if not m.startswith('_')]}")
        model_obj = ModelT()
        model_obj.InitFromObj(root)
        print("### EMBEDDING CONTINUE AFTER INIT FROM OBJ ###")
        # Ensure lists are present (object API uses lowercase field names)
        if getattr(model_obj, 'buffers', None) is None:
            print("### MODEL OBJ BUFFERS EMPTY ###")
            model_obj.buffers = []  # type: ignore[attr-defined]
        if getattr(model_obj, 'metadata', None) is None:
            print("### MODEL METADATA EMPTY ###")
            model_obj.metadata = []  # type: ignore[attr-defined]
        # Check if buffers were actually copied during UnPack
        if hasattr(model_obj, 'buffers') and model_obj.buffers:
            print(f"Unpacked buffers count: {len(model_obj.buffers)}")
            for i, buf in enumerate(model_obj.buffers):
                data_size = len(buf.data) if hasattr(buf, 'data') and buf.data is not None else 0
                print(f"  Buffer {i}: {data_size} bytes")
        else:
            print("WARNING: No buffers found in unpacked model!")
        # Append new buffer with metadata (BufferT.data expects a bytes-like or list of ints)
        new_buf = BufferT()
        print(f"Metadata object type: {type(new_buf)}")
        print(f"Metadata object attributes: {dir(new_buf)}")
        print(f"Metadata buffer size: {len(metadata_buf)} bytes")
        print(f"Metadata buffer type: {type(metadata_buf)}")
        try:
            new_buf.data = bytearray(metadata_buf)  # type: ignore[attr-defined]
            print(f"Successfully set new_buf.data as bytearray, size: {len(new_buf.data)}")
        except Exception as e:
            print("### EXCEPTION EMBEDDING METADATA AS BYTEARRAY ###")
            print(f"EXCEPTION TYPE: {type(e).__name__}")
            print(f"EXCEPTION MESSAGE: {str(e)}")
            print(f"EXCEPTION TRACEBACK: {traceback.format_exc()}")
            new_buf.data = list(metadata_buf)  # type: ignore[attr-defined]
            print(f"Fallback: set new_buf.data as list, size: {len(new_buf.data)}")
        
        print(f"Model object type: {type(model_obj)}")
        print(f"Model object attributes: {dir(model_obj)}")
        
        # Check if model_obj has Buffers attribute
        buffers_attr = getattr(model_obj, 'Buffers', None) or getattr(model_obj, 'buffers', None)
        if buffers_attr is not None:
            buffers_attr.append(new_buf)
            print(f"Appended to buffers, new count: {len(buffers_attr)}")
        else:
            print("WARNING: Could not find buffers attribute on model object")
        # Get buffer index more safely
        buffers_attr = getattr(model_obj, 'Buffers', None) or getattr(model_obj, 'buffers', None)
        if buffers_attr is not None:
            new_buf_index = len(buffers_attr) - 1
            print(f"New buffer index: {new_buf_index}")
        else:
            new_buf_index = 0
            print("WARNING: Using buffer index 0 as fallback")

        # Append metadata entry (MetadataT fields are lowercase)
        meta = MetadataT()
        
        meta.name = "TFLite Metadata"  # type: ignore[attr-defined]
        meta.buffer = new_buf_index  # type: ignore[attr-defined]
        print(f"Set metadata name and buffer index: {new_buf_index}")
        
        # Check if model_obj has metadata attribute
        metadata_attr = getattr(model_obj, 'Metadata', None) or getattr(model_obj, 'metadata', None)
        if metadata_attr is not None:
            metadata_attr.append(meta)
            print(f"Appended to metadata, new count: {len(metadata_attr)}")
        else:
            print("WARNING: Could not find metadata attribute on model object")

        # Pack back to bytes
        print(f"Model object type: {type(model_obj)}")
        print(f"Model buffers count: {len(getattr(model_obj, 'buffers', []))}")
        print(f"Model metadata count: {len(getattr(model_obj, 'metadata', []))}")
        print(f"New buffer data size: {len(getattr(new_buf, 'data', []))}")
        builder = flatbuffers.Builder(0)
        off = model_obj.Pack(builder)
        try:
            print("### BUILDER FINISH TFL3 ###")
            builder.Finish(off, b"TFL3")
        except TypeError:
            print("Exception when finish TFL3")
            builder.Finish(off)
        final_bytes = bytes(builder.Output())
        # After appending metadata, verify it exists
        if hasattr(model_obj, 'metadata') and model_obj.metadata:
            print(f"Metadata entries: {len(model_obj.metadata)}")
        for i, meta in enumerate(model_obj.metadata):
          print(f"  Metadata {i}: name='{meta.name}', buffer={meta.buffer}")
        print(f"Final model size: {len(final_bytes)}")
        print(f"Original vs final size diff: {len(final_bytes) - len(tflite_bytes)}")
        
        # ---- Post-change counts and buffer summary (final model) ----
        try:
            final_root = Model.GetRootAs(final_bytes, 0)
            _collect_model_counts("FINAL", final_root)
            final_buf_sizes = _print_buffer_summary("FINAL", final_root)
            # If buffer counts align, show a quick diff summary
            if orig_buf_sizes and final_buf_sizes and len(orig_buf_sizes) == len(final_buf_sizes):
                changed = [(i, a, b) for i, (a, b) in enumerate(zip(orig_buf_sizes, final_buf_sizes)) if a != b]
                print(f"[DIFF] Buffers changed: {len(changed)} of {len(orig_buf_sizes)}")
                for i, a, b in changed[:10]:
                    print(f"[DIFF]   Buffer {i}: {a} -> {b}")
                if len(changed) > 200:
                    print(f"[DIFF]   ... {len(changed) - 10} more buffer diffs not shown")
            else:
                # If counts differ, highlight the new/last buffer which should be the metadata
                if final_buf_sizes:
                    print(f"[FINAL] Last buffer size (likely metadata): {final_buf_sizes[-1]} bytes")
        except Exception as e:
            print(f"[FINAL] Failed to parse final model for summaries: {e}")
        # Try to parse the final result immediately
        try:
            test_model = Model.GetRootAs(final_bytes, 0)
            print(f"Final model validation: OK, buffers: {test_model.BuffersLength()}")
        except Exception as e:
            print(f"Final model is corrupted: {e}")
        return final_bytes
    except Exception as e:
        print("### EXCEPTION - RETURNING TFLITE_BYTES ###")
        print(f"EXCEPTION TYPE: {type(e).__name__}")
        print(f"EXCEPTION MESSAGE: {str(e)}")
        print(f"EXCEPTION TRACEBACK: {traceback.format_exc()}")
        return tflite_bytes


