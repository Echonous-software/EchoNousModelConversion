import ai_edge_torch
import flatbuffers
import torch
import tensorflow as tf
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
    tensor.description = f'shape: [,{', '.join(str(dim) for dim in io.shape)}]'
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
    metadata.description = "This is the latest description!"
    metadata.version = model.version
    metadata.author = "Echonous, Inc."
    metadata.license = "All rights reserved."
    metadata.minParserVersion = "1.0.0"

    print(f' generating metadata, description: \"{model.description}\"')

    return metadata

def make_metadata_buffer(model: Model) -> bytes:
    metadata = create_metadata(model)
    builder = flatbuffers.Builder(0)
    index = metadata.Pack(builder)
    builder.Finish(index, METADATA_FILE_IDENTIFIER)
    return builder.Output()

def append_metadata_buffer(model_t: _tflite_schema.ModelT, model: Model):
    metadata_bytes = make_metadata_buffer(model)

    # get a buffer index for metadata
    buffer_t = None
    for metadata in model_t.metadata:
        if metadata.name == METADATA_FIELD_NAME:
            if buffer_t is not None:
                raise ValueError(f'Found multiple metadata fields in serialized model file, unsure how to procede')
            # Reuse existing entry, but completely overwrite buffer data
            buffer_t = model_t.buffers[metadata.buffer]
    if buffer_t is None:
        # Create new buffer
        buffer_index = len(model_t.buffers)
        buffer_t = _tflite_schema.BufferT()
        model_t.buffers.append(buffer_t)
        # Add to model list of buffers and save index
        metadata_t = _tflite_schema.MetadataT()
        metadata_t.name = METADATA_FIELD_NAME
        metadata_t.buffer = buffer_index
        model_t.metadata.append(metadata_t)

    buffer_t.data = metadata_bytes

    # also change input/output names
    main_subgraph = model_t.subgraphs[0]
    input_tensors: list[_tflite_schema.TensorT] = [main_subgraph.tensors[i] for i in main_subgraph.inputs]
    output_tensors: list[_tflite_schema.TensorT] = [main_subgraph.tensors[i] for i in main_subgraph.outputs]
    for tensor_t, io in zip(input_tensors, model.inputs):
        print(f' Serialized input tensor: {tensor_t.name} shape: {tensor_t.shape} vs {io.name} shape: {io.shape}')
        #tensor_t.name = io.name
    for tensor_t, io in zip(output_tensors, model.outputs):
        print(f' Serialized output tensor: {tensor_t.name} shape: {tensor_t.shape} vs {io.name} shape: {io.shape}')
        #tensor_t.name = io.name

def save_model_t(model_t: _tflite_schema.ModelT, path: str):
    builder = flatbuffers.Builder(0)
    index = model_t.Pack(builder)
    builder.Finish(index, TFLITE_FILE_IDENTIFIER)
    with open(path, "wb") as f:
        f.write(builder.Output())

def _convert_module_base(model: Model, output_path: str):
    # Use named parameters because tflite seems to ignore input/output order
    module_with_named_parameters = TorchModelWithNames(model)
    module_with_named_parameters.eval()

    # Nothing special in conversion
    sample_inputs = module_with_named_parameters.sample_inputs()
    tflite_model = ai_edge_torch.convert(module_with_named_parameters, sample_kwargs=sample_inputs)
    # export so we can load the flatbuffers schema
    tflite_model.export(output_path)

def _indexof(predicate, iterable) -> int:
    matches = [i for i, item in enumerate(iterable) if predicate(item)]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one item, found {len(matches)} items')
    return matches[0]

def _embed_metadata(model: Model, tflite_filepath: str):
    # read model as ModelT
    with open(tflite_filepath, "rb") as f:
        model_bytes = f.read()
        serialized_model = _tflite_schema.Model.GetRootAs(model_bytes, 0)
        model_t = _tflite_schema.ModelT.InitFromObj(serialized_model)

    # create maps of io name -> tflite_tensor_index
    for subgraphIndex, subgraph in enumerate(model_t.subgraphs):
        subgraph: _tflite_schema.SubGraphT
        input_map = {} # map of input name -> tensor index (in global tensor list)
        output_map = {} # map of output name -> tensor index (in global tensor list)
        signatures = filter(
            lambda signature: signature.subgraphIndex == subgraphIndex,
            model_t.signatureDefs)
        for signature in signatures:
            signature: _tflite_schema.SignatureDefT
            print(f'  signature {signature.signatureKey}')
            for input in signature.inputs:
                if input.name not in input_map:
                    input_map[input.name] = input.tensorIndex
                    if input.tensorIndex not in subgraph.inputs:
                        raise ValueError(f'Input {input.name} maps to tensor index {input.tensorIndex}, but that index is not in subgraph inputs: {subgraph.inputs}')
                elif input_map[input.name] != input.tensorIndex:
                    raise ValueError(f'Input {input.name} maps to multiple tensors in different signatures')
                # tflite_index = subgraph.inputs.index(input.tensorIndex)
                # original_index = _indexof(lambda io: io.name == input.name, model.inputs)
                # print(f'    input {input.name} tflite_index {tflite_index} original_index {original_index}')
            for output in signature.outputs:
                if output.name not in output_map:
                    output_map[output.name] = output.tensorIndex
                    if output.tensorIndex not in subgraph.outputs:
                        raise ValueError(f'Output {output.name} maps to tensor index {output.tensorIndex}, but that index is not in subgraph outputs: {subgraph.inputs}')
                elif output_map[output.name] != output.tensorIndex:
                    raise ValueError(f'Output {output.name} maps to multiple tensors in different signatures')

                # tflite_index = subgraph.outputs.index(output.tensorIndex)
                # original_index = _indexof(lambda io: io.name == output.name, model.outputs)
                # print(f'    output {output.name} tflite_index {tflite_index} original_index {original_index}')

        # Have input_map and output_map, ready to re-order
        # Completely overwrite inputs/outputs lists
        print(f'original subgraph inputs: {subgraph.inputs}')
        subgraph.inputs = [0] * len(model.inputs)
        for idx, input in enumerate(model.inputs):
            if input.name not in input_map:
                raise ValueError(f'Input {input.name} not found in model signature definition(s)')
            tensor_index = input_map[input.name]
            subgraph.inputs[idx] = tensor_index
            subgraph.tensors[tensor_index].name = input.name
        print(f'modified subgraph inputs: {subgraph.inputs}')

        print(f'original subgraph outputs: {subgraph.outputs}')
        subgraph.outputs = [0] * len(model.outputs)
        for idx, output in enumerate(model.outputs):
            if output.name not in output_map:
                raise ValueError(f'Output {output.name} not found in model signature definition(s)')
            tensor_index = output_map[output.name]
            subgraph.outputs[idx] = tensor_index
            subgraph.tensors[tensor_index].name = output.name
        print(f'modified subgraph outputs: {subgraph.outputs}')


    model_t.description = f'{model.description}\n{"-"*50}\n'


    # create new ModelMetadataT
    append_metadata_buffer(model_t, model)
    save_model_t(model_t, tflite_filepath)
    # append/overwrite metadata buffer in ModelT

def convert_model(model: Model, output_path: str):
    # convert to tflite model
    _convert_module_base(model, output_path)

    # embed metadata and re-save file
    _embed_metadata(model, output_path)
        # read model as ModelT
        # read metadata from ModelT buffer
        # match input names in Model to input names in metadata
            # produces map of { input_name: tflite_index }
        # match output names in Model to output names in metadata
            # produces map of { output_name: tflite_index }
        # reorder tflite inputs/outputs to match model

        # create new ModelMetadataT
        # append/overwrite metadata buffer in ModelT

    # resave file

def main():
    from echonous.models.loaders import load_model
    model = load_model('guidance.psax_av')

    convert_model(model, "guidance_psax_av2.tflite")
    return



    model.module.eval()

    named_model = TorchModelWithNames(model)
    named_model.eval()

    inputs = sample_inputs(model)
    print(inputs)
    tflite_model = ai_edge_torch.convert(named_model, sample_kwargs=inputs)
    print(tflite_model)
    tflite_model.export('guidance_psax_av.tflite')
    with open('guidance_psax_av.tflite', 'rb') as f:
        model_bytes = f.read()

    # Load the model
    interpreter = tf.lite.Interpreter(model_path="guidance_psax_av.tflite")
    interpreter.allocate_tensors()

    # Get output details
    output_details = interpreter.get_output_details()
    for i, output in enumerate(output_details):
        print(f"Output {i}: {output['name']}, shape: {output['shape']}")

    serialized_model = _tflite_schema.Model.GetRootAs(model_bytes, 0)
    model_t = _tflite_schema.ModelT.InitFromObj(serialized_model)

    append_metadata_buffer(model_t, model)
    save_model_t(model_t, f'guidance_psax_av_with_metadata.tflite')

if __name__ == '__main__':
    main()
