Introduction
=============

This library automatically extracts intermediate outputs from a pytorch model.

Installation/Usage:
*******************
To install directly from source as an editable module, run these commands from the terminal:

.. code-block:: bash
    git clone git@github.com:edithvillegas/activation-extractor.git
    cd activation-extractor
    python -m pip install -e .

The source code can be found `here <https://github.com/edithvillegas/activation-extractor/>`_

Quick start:
**************************************************

.. code-block:: python
    import activation_extractor

    #load model
    model_name = "facebook/esm2_t6_8M_UR50D" 
    inferencer = activation_extractor.Inferencer(model_name, device='cpu', half=False)
    
    #load data
    sequences = ["AAAAAAAAAAA", "HHHHHHHHHHHHHH"]
    
    #intermediate activation extractor
    layers_to_hook = activation_extractor.get_layers_to_hook(inferencer.model,inferencer.model_type)
    extractor = activation_extractor.IntermediateExtractor(inferencer.model, layers_to_hook)
    extractor.register_hooks()
    
    #inference
    tokens_ids = inferencer.tokenize(sequences)
    outputs = inferencer.inference(tokens_ids)
    
    #extractor outputs
    intermediate_outputs = extractor.get_outputs()

    #close extractor
    extractor.clear_all_hooks()
