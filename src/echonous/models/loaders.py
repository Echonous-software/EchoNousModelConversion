import textwrap
from importlib import resources

import torch
import yaml

from echonous.models import Model, ModuleInputOutput

def main():
    catalog_path = resources.files('echonous.models') / 'catalog.yaml'
    with catalog_path.open('r') as f:
        catalog = yaml.safe_load(f)

    models = {}

    for name, params in catalog['models'].items():
        match params['model_loader']:
            case 'babajide_guidance_model':
                models[name] = load_babajide_guidance_model(name, params)

    for name, model in models.items():
        print(f'Model {name}:')
        print(f'  internal name: {model.name}')
        print(f'  version:       {model.version}')
        print(f'  description:')
        print(textwrap.indent(model.description, '    ').rstrip())
        print(f'  inputs:')
        for input in model.inputs:
            print(f'    {input.name} is {input.shape} type {input.type} scale {input.scale}')
        print(f'  outputs:')
        for output in model.outputs:
            print(f'    {output.name} is {output.shape} type {output.type} scale {output.scale}')

def load_babajide_guidance_model(name: str, params: dict) -> Model:
    from echonous._vendor.babajide_guidance.models.mobilenet2out import Net1out
    from echonous._vendor.babajide_guidance.view_json import json_view as babajide_config

    view = params['view']
    view_config = babajide_config[view]

    # Get model hyperparameters and other values which vary by type
    match params['type']:
        case 'quality':
            cb = [16, 32, 64, 128, 256]
            out_dim = view_config['model_param'][1]
            weights_name = view_config['IMAGEQUALITY_model_name']
            output = 'quality'
        case 'subview':
            cb = [32, 64, 128, 256, 512]
            out_dim = view_config['model_param'][0]
            weights_name = view_config['SUBVIEW_model_name']
            output = 'subview'
        case _:
            raise KeyError(f"Unsupported model type {params['type']}, must be either quality or subview")

    # Create pytorch module and load the weights
    model = Net1out(cb=cb, out_dim=out_dim, inp_ch=1)
    weights_path = resources.files(f'echonous._vendor.babajide_guidance.weights.{view}') / weights_name
    with weights_path.open('rb') as f:
        weights = torch.load(f, map_location='cpu', weights_only=True)
    model.load_state_dict(weights)
    model.eval()

    # Model metadata to embed in converted files
    # Probably shouldn't use '.' characters in exported model name, or spaces
    model_name = name.replace('.', '_')
    model_version = _get_model_version('babajide_guidance', params['version'])
    model_description = _get_model_description('babajide_guidance')

    image_size = params['image_size']
    inputs = [ModuleInputOutput(
        name='image',
        shape=(1, 1, image_size, image_size),
        type='image',
        scale=1.0
    )]
    outputs = [ModuleInputOutput(
        name=output,
        shape=(1, out_dim)
    )]
    return Model(
        name=model_name,
        version=model_version,
        description=model_description,
        module=model,
        inputs=inputs,
        outputs=outputs
    )


def _get_model_version(vendor_name: str, version: str | dict) -> str:
    if isinstance(version, str):
        return version
    elif version['type'] == 'git':
        # use version number found in repo.yaml file
        repo_metadata_path = resources.files(f'echonous._vendor.{vendor_name}') / 'repo.yaml'
        with repo_metadata_path.open('r') as f:
            repo_metadata = yaml.safe_load(f)
        return repo_metadata['refspec']
    else:
        raise ValueError(f"Unknown version spec: {version}")


def _get_model_description(vendor_name: str) -> str:
    repo_metadata_path = resources.files(f'echonous._vendor.{vendor_name}') / 'repo.yaml'
    with repo_metadata_path.open('r') as f:
        content = f.read()
    return content

if __name__ == '__main__':
    main()