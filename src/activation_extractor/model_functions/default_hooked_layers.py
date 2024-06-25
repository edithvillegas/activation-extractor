"""
This file contains functions to get relevant layer (module) names to hook from the models included by default.
"""

def get_layers_to_hook(model, model_type):
    """
    Get a list of default layers to hook (extract activations from) for each model type.

    :param model: the Pytorch model object
    :param model_type: A model type (protein - esm, prot_t5, ankh; dna - nucleotide-transformer, hyenadna, evo, caduceus).
    :type model_type: str

    :return: the list of layers/modules names
    :rtype: list
    """
    
    match model_type:
        # Protein Sequence & DNA 🥩, 🧬
        case "esm" | "nucleotide-transformer": 
            n_layers = model.config.num_hidden_layers
            layers_to_hook = ["esm.embeddings", "lm_head"] + [f"esm.encoder.layer.{n}.output" for n in range(0,n_layers)]
            
        # DNA Models 🧬
        case "hyenadna":
            n_layers = model.config.n_layer
            layers_to_hook = ["backbone.embeddings"] + [f"backbone.layers.{n}" for n in range(n_layers)]
            
        case "evo":
            n_layers = model.config.num_layers
            layers_to_hook = [f"blocks.{n}" for n in range(n_layers)] # + ["embedding_layer", "unembed"]
            
        case "caduceus":
            n_layers = model.config.n_layer
            layers_to_hook = [f"caduceus.backbone.layers.{n}.mixer" for n in range(n_layers)] # + ["embedding_layer", "unembed"]
            
            
        # Protein Sequence Models 🥩
        case "prot_t5":
            n_layers = model.config.num_decoder_layers
            layers_to_hook = [f"encoder.block.{n}" for n in range(n_layers)] # ["encoder.embed_tokens"] 
            
        case "prot_bert":
            n_layers = model.config.num_hidden_layers
            layers_to_hook = ["embeddings"]+[f"bert.encoder.layer.{n}" for n in range(n_layers)]
            
        case "prot_xlnet":
            n_layers = model.config.num_hidden_layers
            layers_to_hook = ["word_embedding"]+[f"layer.{n}" for n in range(n_layers)]
            
        case "prot_electra":
            n_layers = model.config.num_hidden_layers
            layers_to_hook = ["embeddings"]+[f"encoder.layer.{n}" for n in range(n_layers)]
            
        case "ankh":
            n_layers = model.config.num_layers
            layers_to_hook = [f"encoder.block.{n}" for n in range(n_layers)]
            
        case "prostt5":
            n_layers = model.config.num_layers
            layers_to_hook = [f"encoder.block.{n}" for n in range(n_layers)]
        

        # Image Models 🖼️
        case "vit":
            n_layers = model.config.num_hidden_layers
            layers_to_hook = [f"vit.encoder.layer.{n}" for n in range(n_layers)] 

        case "igpt":
            n_layers = model.config.n_layer
            layers_to_hook = [f"h.{n}" for n in range(n_layers)]

        case "swin":
            layers_to_hook = []
            for ns in range( len(model.config.depths) ):
                for nl in range( model.config.depths[ns] ):
                    layers_to_hook += [f"swin.encoder.layers.{ns}.blocks.{nl}"]

        case "convnext":
            layers_to_hook = []
            for ns in range(model.config.num_stages):
                layers_to_hook += [f"convnext.encoder.stages.{ns}.layers.{nl}" for nl in range(model.config.depths[ns])]

        case "resnet":
            layers_to_hook = []
            for ns in range( len(model.config.depths) ):
                for nl1 in range( model.config.depths[ns] ):
                    for nl2 in range(0,3):
                        layers_to_hook += [f"resnet.encoder.stages.{ns}.layers.{nl1}.layer.{nl2}"] 

        case "timm":
            layers_to_hook = []
            for name, module in model.named_modules():
                layers_to_hook.append(name)

        #multimodal 🖼️/📚
        case "clip":
            layers_to_hook = (["text_model.embeddings", "vision_model.embeddings"]
                            + [f"text_model.encoder.layers.{n}" for n in range(model.text_model.config.num_hidden_layers)]
                            + [f"vision_model.encoder.layers.{n}" for n in range(model.vision_model.config.num_hidden_layers)]
                            + ["visual_projection", "text_projection"])
        #default
        case _:
            raise ValueError(f"model_type not valid")
    
    return layers_to_hook
