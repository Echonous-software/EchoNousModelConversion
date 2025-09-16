import json
from dataclasses import asdict

from echonous.exporters import ai_edge_exporter
from echonous.exporters._tflite_schema import tflite_metadata_generated as _tflite_metadata
from echonous.models import ModuleInputOutput
from tests.exporters.sample_models import MultiIOModel

def _assert_io_matches(io: ModuleInputOutput, metadata: _tflite_metadata.TensorMetadataT) -> None:
    """Assert that an I/O definition matches a serialized tflite tensor metadata."""
    assert metadata.name == io.name
    #
    description = json.loads(metadata.description)
    expected_description = asdict(io)
    # Shape in `io` is a tuple, but json serialization will only ever give us a list
    # so convert the expected type to a list as well
    expected_description["shape"] = list(expected_description["shape"])
    assert description == expected_description

def test_create_tensor_metadata():
    io = ModuleInputOutput(name="sample_input", shape=(1,1,224,224), type='image', scale=1.0/255.0)
    metadata = ai_edge_exporter.create_tensor_metadata(io)
    _assert_io_matches(io, metadata)

def test_create_subgraph_metadata():
    model = MultiIOModel().definition()
    metadata = ai_edge_exporter.create_subgraph_metadata(model)
    assert metadata.name == model.name
    assert metadata.description == model.description
    for io, tensor_metadata in zip(model.inputs, metadata.inputTensorMetadata, strict=True):
        _assert_io_matches(io, tensor_metadata)
    for io, tensor_metadata in zip(model.outputs, metadata.outputTensorMetadata, strict=True):
        _assert_io_matches(io, tensor_metadata)

