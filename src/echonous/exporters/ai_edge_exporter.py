import ai_edge_torch
import torch

from echonous.models import Model


def sample_inputs(model: Model):
    return tuple(
        torch.rand(io.shape) for io in model.inputs
    )

def main():
    from echonous.models.loaders import load_model
    model = load_model('guidance.psax_av')
    inputs = sample_inputs(model)
    print(inputs)
    tflite_model = ai_edge_torch.convert(model.module, inputs)
    print(tflite_model)
    tflite_model.export('guidance_psax_av.tflite')

if __name__ == '__main__':
    main()
