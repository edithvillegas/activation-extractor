# Activation Extractor
Extracting neuron activations/embeddings from the intermediate layers of any Pytorch model.

## Installation
To install directly from source as an editable module, run these commands from the terminal:

```
git clone git@github.com:edithvillegas/activation-extractor.git
cd activation-extractor
python -m pip install -e .
```

## Sample Usage 

1. Load the package.

```python
import activation_extractor
```

2. Load the pytorch model. Some models are included in the package and can be loaded directly, otherwise, load it the usual way.
```python
model_name = "facebook/esm2_t6_8M_UR50D" 
inferencer = activation_extractor.Inferencer(model_name, device='cpu', half=False)
```

3. Load the data.
```python
sequences = ["AAAAAAAAAAA", "HHHHHHHHHHHHHH"]
```

4. Initialize the activation extractor.
The first argument to the extractor initialization is the model object (```inferencer.model```).
The ```layers_to_hook``` variable should be a list with all the layers/modules' names that you want to get the activations from.
For the models included in the library by default, the function ```get_layers_to_hook``` returns a list of all the most relevant layers from the model.

```python
layers_to_hook = activation_extractor.get_layers_to_hook(inferencer.model,inferencer.model_type)
extractor = activation_extractor.IntermediateExtractor(inferencer.model, layers_to_hook)
extractor.register_hooks()
```

5. Perform inference as usual.
```python
tokens_ids = inferencer.tokenize(sequences)
outputs = inferencer.inference(tokens_ids)
```

6. Look at the outputs.
```python
intermediate_outputs = extractor.get_outputs()
```
