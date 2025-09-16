import json
from dataclasses import asdict
from typing import Literal

import ai_edge_torch
import flatbuffers
import torch
from torch import nn

from echonous.models import Model, ModuleInputOutput
import echonous.exporters._tflite_schema.tflite_generated as _tflite_schema
import echonous.exporters._tflite_schema.tflite_metadata_generated as _tflite_metadata

METADATA_FIELD_NAME = "TFLITE_METADATA"
TFLITE_FILE_IDENTIFIER = b"TFL3"
METADATA_FILE_IDENTIFIER = b"M001"

class TorchModelWithNames(nn.Module):
    """Wraps a torch module and uses maps of names -> tensors for input/output"""
    def __init__(self, model: Model):
        super().__init__()
        self._module = model.module
        self._inputs = model.inputs
        self._outputs = model.outputs

    def forward(self, **kwargs):
        input_list = [kwargs[io.name] for io in self._inputs]
        output_list = self._module(*input_list)
        if not isinstance(output_list, (list, tuple)):
            output_list = [output_list]
        return {
            io.name: output for io, output in zip(self._outputs, output_list)
        }

    def sample_inputs(self):
        return {
            io.name: torch.rand(io.shape) for io in self._inputs
        }

def create_tensor_metadata(io: ModuleInputOutput) -> _tflite_metadata.TensorMetadataT:
    tensor = _tflite_metadata.TensorMetadataT()
    tensor.name = io.name
    tensor.description = json.dumps(asdict(io))
    return tensor


def create_subgraph_metadata(model: Model) -> _tflite_metadata.SubGraphMetadataT:
    subgraph = _tflite_metadata.SubGraphMetadataT()
    subgraph.name = model.name
    subgraph.description = model.description
    subgraph.inputTensorMetadata = [create_tensor_metadata(input) for input in model.inputs]
    subgraph.outputTensorMetadata = [create_tensor_metadata(output) for output in model.outputs]
    # potential todo: add associated files
    return subgraph

def create_metadata(model: Model) -> _tflite_metadata.ModelMetadataT:
    metadata = _tflite_metadata.ModelMetadataT()
    metadata.name = model.name
    metadata.description = model.description
    metadata.version = model.version
    metadata.author = "Echonous, Inc."
    metadata.license = "All rights reserved."
    metadata.minParserVersion = "1.0.0"
    metadata.subgraphMetadata = [create_subgraph_metadata(model)]

    print(f' generating metadata, description: \"{model.description}\"')

    return metadata

def make_metadata_buffer(model: Model) -> bytes:
    metadata = create_metadata(model)
    builder = flatbuffers.Builder(0)
    index = metadata.Pack(builder)
    builder.Finish(index, METADATA_FILE_IDENTIFIER)
    return builder.Output()

def get_buffer_for_metadata(model_t: _tflite_schema.ModelT) -> _tflite_schema.BufferT:
    """
    Get a BufferT suitable for holding model metadata (serialized bytes of a ModelMetadataT).

    A ModelT has a list of metadata entries. At most one of those entries should have a name of
    METADATA_FIELD_NAME ("TFLITE_METADATA"). This function finds and returns the buffer identified
    by this name, or creates one if needed.
    """
    # Find existing metadata buffer
    metadata_entries = [m for m in model_t.metadata if m.name == METADATA_FIELD_NAME]
    
    if len(metadata_entries) > 1:
        raise ValueError("Found multiple metadata fields in model file, this is probably a bug")
    
    if metadata_entries:
        return model_t.buffers[metadata_entries[0].buffer]
    
    # Create new metadata buffer
    buffer_t = _tflite_schema.BufferT()
    model_t.buffers.append(buffer_t)
    metadata = _tflite_schema.MetadataT()
    metadata.name = METADATA_FIELD_NAME
    metadata.buffer = len(model_t.buffers) - 1
    model_t.metadata.append(metadata)
    
    return buffer_t


def append_metadata_buffer(model_t: _tflite_schema.ModelT, model: Model) -> None:
    metadata_bytes = make_metadata_buffer(model)
    buffer_t = get_buffer_for_metadata(model_t)
    buffer_t.data = metadata_bytes


def save_model_t(model_t: _tflite_schema.ModelT, path: str) -> None:
    builder = flatbuffers.Builder(0)
    index = model_t.Pack(builder)
    builder.Finish(index, TFLITE_FILE_IDENTIFIER)
    with open(path, "wb") as f:
        f.write(builder.Output())

def _convert_module_base(model: Model, output_path: str) -> None:
    # Use named parameters because tflite seems to ignore input/output order
    module_with_named_parameters = TorchModelWithNames(model)
    module_with_named_parameters.eval()

    # Nothing special in conversion
    sample_inputs = module_with_named_parameters.sample_inputs()
    tflite_model = ai_edge_torch.convert(module_with_named_parameters, sample_kwargs=sample_inputs)
    # export so we can load the flatbuffers schema
    tflite_model.export(output_path)


def _load_tflite_model(tflite_filepath: str) -> _tflite_schema.ModelT:
    """Load and deserialize a TFLite model file."""
    with open(tflite_filepath, "rb") as f:
        model_bytes = f.read()
        serialized_model = _tflite_schema.Model.GetRootAs(model_bytes, 0)
        return _tflite_schema.ModelT.InitFromObj(serialized_model)

def _process_signature_io(signature_io_list: list[_tflite_schema.TensorMapT], io_map: dict[str, int], subgraph: _tflite_schema.SubGraphT, io_type: str) -> None:
    """Process input or output tensors from signatures, building the name-to-tensor-index map."""
    subgraph_io_list = subgraph.inputs if io_type == "input" else subgraph.outputs
    
    for io_item in signature_io_list:
        if io_item.name not in io_map:
            io_map[io_item.name] = io_item.tensorIndex
            if io_item.tensorIndex not in subgraph_io_list:
                raise ValueError(f'{io_type.capitalize()} {io_item.name} maps to tensor index {io_item.tensorIndex}, '
                               f'but that index is not in subgraph {io_type}s: {subgraph_io_list}')
        elif io_map[io_item.name] != io_item.tensorIndex:
            raise ValueError(f'{io_type.capitalize()} {io_item.name} maps to multiple tensors in different signatures')

def _build_signature_maps(model_t: _tflite_schema.ModelT, subgraph_index: int) -> tuple[dict[str, int], dict[str, int]]:
    """Build maps of input/output names to tensor indices from signature definitions."""
    input_map = {}  # map of input name -> tensor index (in global tensor list)
    output_map = {}  # map of output name -> tensor index (in global tensor list)
    subgraph = model_t.subgraphs[subgraph_index]
    
    signatures = filter(
        lambda signature: signature.subgraphIndex == subgraph_index,
        model_t.signatureDefs)
    
    for signature in signatures:
        print(f'  signature {signature.signatureKey}')
        _process_signature_io(signature.inputs, input_map, subgraph, "input")
        _process_signature_io(signature.outputs, output_map, subgraph, "output")
    
    return input_map, output_map

def _reorder_subgraph_io(subgraph: _tflite_schema.SubGraphT, model_io_list: list[ModuleInputOutput], io_map: dict[str, int], io_type: Literal["input", "output"]) -> None:
    """Reorder subgraph inputs or outputs to match the original model order."""
    current_list = subgraph.inputs if io_type == "input" else subgraph.outputs
    print(f'original subgraph {io_type}s: {current_list}')
    
    new_io_list = [0] * len(model_io_list)
    for idx, io_item in enumerate(model_io_list):
        if io_item.name not in io_map:
            raise ValueError(f'{io_type.capitalize()} {io_item.name} not found in model signature definition(s)')
        tensor_index = io_map[io_item.name]
        new_io_list[idx] = tensor_index
        subgraph.tensors[tensor_index].name = io_item.name
    
    if io_type == "input":
        subgraph.inputs = new_io_list
    else:
        subgraph.outputs = new_io_list
    print(f'modified subgraph {io_type}s: {new_io_list}')

def _finalize_model(model_t: _tflite_schema.ModelT, model: Model, tflite_filepath: str) -> None:
    """Add description, metadata, and save the final model."""
    model_t.description = f'{model.description}\n{"-"*50}\n'
    append_metadata_buffer(model_t, model)
    save_model_t(model_t, tflite_filepath)

def _embed_metadata(model: Model, tflite_filepath: str) -> None:
    """Embed metadata into a TFLite model file by reordering inputs/outputs and adding metadata."""
    # Load the TFLite model
    model_t = _load_tflite_model(tflite_filepath)

    # Process each subgraph
    for subgraph_index, subgraph in enumerate(model_t.subgraphs):
        # Build maps of input/output names to tensor indices from signatures
        input_map, output_map = _build_signature_maps(model_t, subgraph_index)
        
        # Reorder inputs and outputs to match the original model order
        _reorder_subgraph_io(subgraph, model.inputs, input_map, "input")
        _reorder_subgraph_io(subgraph, model.outputs, output_map, "output")

    # Finalize the model with description, metadata, and save
    _finalize_model(model_t, model, tflite_filepath)

def convert_model(model: Model, output_path: str) -> None:
    # convert to tflite model
    _convert_module_base(model, output_path)
    # embed metadata and re-save file
    _embed_metadata(model, output_path)


def main() -> None:
    from echonous.models.loaders import load_model
    model = load_model('guidance.psax_av')
    convert_model(model, "guidance_psax_av2.tflite")


if __name__ == '__main__':
    main()
