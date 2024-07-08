"""
This file contains functions to get relevant layer (module) names to hook from the models included by default.
"""

def get_layers_to_hook(model, model_type, modality="sequence", return_structure=False):
    """
    Get a list of default layers to hook (extract activations from) for each model type.

    :param model: the Pytorch model object
    :param model_type: A model type (protein - esm, prot_t5, ankh; dna - nucleotide-transformer, hyenadna, evo, caduceus).
    :type model_type: str

    :return: the list of layers/modules names
    :rtype: list
    """
    ### Sequence Models ####
    match model_type:
        # Protein Sequence & DNA 🥩, 🧬
        case "esm" | "nucleotide-transformer": 
            n_layers = model.config.num_hidden_layers
            embeddings = ["esm.embeddings"]
            layers = [f"esm.encoder.layer.{n}.output" for n in range(0,n_layers)]
            
        # DNA Models 🧬
        case "hyenadna":
            n_layers = model.config.n_layer
            embeddings = ["backbone.embeddings"]
            layers = [f"backbone.layers.{n}" for n in range(n_layers)]
            
        case "evo":
            n_layers = model.config.num_layers
            embeddings = [] #❗
            layers = [f"blocks.{n}" for n in range(n_layers)] # + ["embedding_layer", "unembed"]
            
        case "caduceus":
            n_layers = model.config.n_layer
            embeddings = [] #❗
            layers = [f"caduceus.backbone.layers.{n}.mixer" for n in range(n_layers)] # + ["embedding_layer", "unembed"]
            
        # Protein Sequence Models 🥩
        case "prot_t5":
            n_layers = model.config.num_decoder_layers
            embeddings = ["shared"]
            layers = [f"encoder.block.{n}" for n in range(n_layers)] 
            
        case "prot_bert":
            n_layers = model.config.num_hidden_layers
            embeddings = ["bert.embeddings"]
            layers = [f"bert.encoder.layer.{n}" for n in range(n_layers)]
            
        case "prot_xlnet":
            n_layers = model.config.num_hidden_layers
            embeddings = ["word_embedding"]
            layers = [f"layer.{n}" for n in range(n_layers)]
            
        case "prot_electra":
            n_layers = model.config.num_hidden_layers
            embeddings = ["embeddings"]
            layers = [f"encoder.layer.{n}" for n in range(n_layers)]
            
        case "ankh":
            n_layers = model.config.num_layers
            embeddings = ["shared"]
            layers = embeddings + [f"encoder.block.{n}" for n in range(n_layers)]

        # Protein sequence / 3di 🥩/⛓️
        case "prostt5":
            n_layers = model.config.num_layers
            embeddings = ["shared"]
            layers = [f"encoder.block.{n}" for n in range(n_layers)]

        # Protein sequence / structure / function 🥩/🧱/🌟
        case "esm3":
            embeddings = ["encoder.sequence_embed"]
            layers = (
                ["transformer.blocks.0.attn", "transformer.blocks.0.geom_attn"]
                +[f"transformer.blocks.{n}" for n in range(47+1)]
            )

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

        case "visual-mamba":
            pass
            
        #text 📚
        case "pythia":
            n_layers = model.config.num_hidden_layers
            embeddings = ["gpt_neox.embed_in"]
            layers = [f"gpt_neox.layers.{n}" for n in range(n_layers)]

        case "mamba":
            n_layers = model.config.n_layer
            embeddings = ["backbone.embeddings"]
            layers = [f"backbone.layers.{n}" for n in range(n_layers)]
            
        #multimodal 🖼️/📚
        case "clip":
            text_embeddings = ["text_model.embeddings"]
            image_embeddings = ["vision_model.embeddings"]
            text_layers = (text_embeddings 
                           + [f"text_model.encoder.layers.{n}" for n in range(model.text_model.config.num_hidden_layers)]
                           + ["text_projection"])
            image_layers = (image_embeddings 
                           + [f"vision_model.encoder.layers.{n}" for n in range(model.text_model.config.num_hidden_layers)]
                           + ["visual_projection"])
            
        #default
        case _:
            raise ValueError(f"model_type not valid")

    #construct structure 
    match modality:
        case "sequence":
            layers_to_hook = embeddings + layers
            
            structure = {
                "embeddings": embeddings,
                "layers": layers,
            }
            
        case "image-text":
            layers_to_hook = text_layers + image_layers 
            structure = {
                "text_embeddings": text_embeddings,
                "text_layers": text_layers,
                "image_embeddings": image_embeddings,
                "image_layers": image_layers,
            }
        
        case _:
            structure = {
                "layers": layers_to_hook,
            }

    #return
    if return_structure==False:
        return layers_to_hook
    else:
        return layers_to_hook, structure
