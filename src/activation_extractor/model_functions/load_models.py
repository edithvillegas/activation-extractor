"""
This file contains functions to load models, tokenizers, etc.
Because not all models are from huggingface and they might not all be installed, 
the right imports are directly inside the corresponding loading functions. 
"""
# load tokenizers ⏳
def load_tokenizer(model_name, tokenizer_type, **kwargs):
    """
    Load a tokenizer type for a model. 
    This function is called inside load_model() for sequence type models.

    :param model_name: model name (for huggingface models it should be the same as the loaded model)
    :type model_name: str
    :param tokenizer_type: the type of tokenizer (valid types - AutoTokenizer and T5Tokenizer)
    :type tokenizer_type: str

    :return: the tokenizer object
    """
    match tokenizer_type:
        case "AutoTokenizer":
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, **kwargs)
        case "T5Tokenizer":
            from transformers import T5Tokenizer
            tokenizer = T5Tokenizer.from_pretrained(model_name, trust_remote_code=True, do_lower_case=False, **kwargs)
        case _:
            raise ValueError(f"tokenizer_type not valid")
    return tokenizer

# load models ⏳
def load_model(model_name, model_type, **kwargs):
    """
    Get a list of default layers to hook for each model type.

    :param model: the Pytorch model object
    :param model_type: A model type (protein - esm, prot_t5, ankh; dna - nucleotide-transformer, hyenadna, evo, caduceus).
    :type model_type: str

    :return: the list of layers/modules names
    :rtype: list
    """
    match model_type:
    #DNA models 🧬 --- 
        case "nucleotide-transformer":
            from transformers import AutoModelForMaskedLM
            model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True, **kwargs)
            tokenizer = load_tokenizer(model_name, tokenizer_type='AutoTokenizer', **kwargs)
            return model, tokenizer
            
        case 'hyenadna':
            from transformers import AutoModel
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True, **kwargs)
            #model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True, **kwargs)   
            tokenizer = load_tokenizer(model_name, tokenizer_type='AutoTokenizer', **kwargs)
            return model, tokenizer
            
        case 'evo':
            from evo import Evo
            evo_model = Evo(model_name.split('/')[1])
            model, tokenizer = evo_model.model, evo_model.tokenizer
            return model, tokenizer
            
        case 'caduceus':
            from transformers import AutoModelForMaskedLM
            #model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **kwargs)
            model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True, **kwargs)
            tokenizer = load_tokenizer(model_name, tokenizer_type='AutoTokenizer', **kwargs)
            return model, tokenizer
            
    #protein models 🥩 ---
        case 'esm':
            from transformers import AutoModelForMaskedLM
            model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True, **kwargs)
            tokenizer = load_tokenizer(model_name, tokenizer_type='AutoTokenizer', **kwargs)
            return model, tokenizer
            
        case 'prot_t5':
            from transformers import T5EncoderModel
            model = T5EncoderModel.from_pretrained(model_name, **kwargs)
            tokenizer = load_tokenizer(model_name, tokenizer_type='T5Tokenizer', **kwargs)
            return model, tokenizer
            
        case 'ankh':
            from transformers import T5EncoderModel
            #model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **kwargs)
            model = T5EncoderModel.from_pretrained(model_name, **kwargs) #output_attentions
            tokenizer = load_tokenizer(model_name, tokenizer_type='AutoTokenizer', **kwargs)
            return model, tokenizer

        case _:
            raise ValueError(f"model_type not valid ")
    #----
    