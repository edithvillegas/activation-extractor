"""
This file contains functions to load models, tokenizers, etc.
Because not all models are from huggingface and they might not all be installed, 
the right imports are directly inside the corresponding loading functions. 
"""
import os

#⏳ load tokenizers 
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
        case "BertTokenizer":
            from transformers import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained(model_name, **kwargs)
        case _:
            raise ValueError(f"tokenizer_type not valid")
    return tokenizer

#⏳ load processors
def load_processor(model_name, processor_type, **kwargs):
    match processor_type:
        #images 🖼️
        case "AutoProcessor":
            from transformers import AutoImageProcessor
            processor = AutoImageProcessor.from_pretrained(model_name)

        case "igptProcessor":
            from transformers import ImageGPTImageProcessor, ImageGPTModel
            processor = ImageGPTImageProcessor.from_pretrained(model_name)

        case "convnextProcessor":
            from transformers import ConvNextImageProcessor
            processor = ConvNextImageProcessor.from_pretrained(model_name)
            
        #multimodal 🖼️/📚 
        case "CLIP":
            from transformers import CLIPProcessor
            processor = CLIPProcessor.from_pretrained(model_name)
            
    return processor

#⏳ load models
def load_model(model_name, model_type, **kwargs):
    """
    Loads a Pytorch model according to the passed model name. 
    For sequence models, it loads the corresponding tokenizer.
    For image models, it loads the image processor.

    :param model: the Pytorch model object
    :param model_type: A model type (see list of included models).
    :type model_type: str

    :return: tuple with (model, tokenizer) or (model, processor).  
    """
    #START OF MATCH#
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
            
        case 'prot_t5' | "prostt5" :
            from transformers import T5EncoderModel
            model = T5EncoderModel.from_pretrained(model_name, **kwargs)
            tokenizer = load_tokenizer(model_name, tokenizer_type='T5Tokenizer', **kwargs)
            return model, tokenizer

        case "prot_bert":
            from transformers import BertForMaskedLM
            model = BertForMaskedLM.from_pretrained(model_name, **kwargs)
            tokenizer = load_tokenizer(model_name, tokenizer_type="BertTokenizer", 
                                       do_lower_case=False, **kwargs)
            return model, tokenizer

        case "prot_xlnet":
            from transformers import XLNetTokenizer, XLNetModel
            model = XLNetModel.from_pretrained(model_name, 
                                               output_attentions=False)
            tokenizer = XLNetTokenizer.from_pretrained(model_name, 
                                                       do_lower_case=False)
            return model, tokenizer

        case "prot_electra":
            from transformers import (ElectraTokenizer, 
                                        ElectraForMaskedLM, ElectraModel, 
                                        AutoModel)
            from activation_extractor.utils.download import download_file

            #crate folder to download models
            FolderPath = f"{os.environ['TRANSFORMERS_CACHE']}/electra/{model_name}"
            os.makedirs(FolderPath, exist_ok=True)

            # #corresponding urls for each model
            # if model_name=="Rostlab/prot_electra_generator_bfd":
            #     ModelUrl = 'https://www.dropbox.com/s/5x5et5q84y3r01m/pytorch_model.bin?dl=1'
            #     ConfigUrl = 'https://www.dropbox.com/s/9059fvix18i6why/config.json?dl=1'
                
            # if model_name=="Rostlab/prot_electra_discriminator_bfd":
            #     ModelUrl = 'https://www.dropbox.com/s/9ptrgtc8ranf0pa/pytorch_model.bin?dl=1'
            #     ConfigUrl = 'https://www.dropbox.com/s/jq568evzexyla0p/config.json?dl=1'

            # #download files
            # ModelFilePath = os.path.join(FolderPath, 'pytorch_model.bin')
            # ConfigFilePath = os.path.join(FolderPath, 'config.json')
            # download_file(ModelUrl, ModelFilePath)
            # download_file(ConfigUrl, ConfigFilePath)

            # #create model
            # # 
            # if model_name=="Rostlab/prot_electra_generator_bfd":
            #     model = ElectraForMaskedLM.from_pretrained(FolderPath, output_attentions=False)
                
            # if model_name=="Rostlab/prot_electra_discriminator_bfd":
            #     model = ElectraModel.from_pretrained(FolderPath, output_attentions=False)

            model = AutoModel.from_pretrained(model_name)
            
            #tokenizer
            # vocabUrl = 'https://www.dropbox.com/s/wck3w1q15bc53s0/vocab.txt?dl=1'
            # vocabFilePath = f"{FolderPath}/vocab.txt"
            # download_file(vocabUrl, vocabFilePath)
            
            # tokenizer = ElectraTokenizer(vocabFilePath, do_lower_case=False)
            
            #using Prot_Bert tokenizer ❗
            tokenizer = load_tokenizer("Rostlab/prot_bert", 
                                       tokenizer_type='AutoTokenizer', 
                                       **kwargs)
            return model, tokenizer
            
        case 'ankh':
            from transformers import T5EncoderModel
            #model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **kwargs)
            model = T5EncoderModel.from_pretrained(model_name, **kwargs) #output_attentions
            tokenizer = load_tokenizer(model_name, tokenizer_type='AutoTokenizer', **kwargs)
            return model, tokenizer

        case "esm3": #🥩/🧱/🌟
            from esm.models.esm3 import ESM3
            from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
            
            model: ESM3InferenceClient = ESM3.from_pretrained("esm3_sm_open_v1")
            tokenizer = EsmSequenceTokenizer()
            return model, tokenizer

        #image 🖼️ --- 
        case "vit":
            from transformers import ViTForMaskedImageModeling
            processor = load_processor(model_name, processor_type="AutoProcessor")
            model = ViTForMaskedImageModeling.from_pretrained(model_name)
            return model, processor

        case "igpt":
            from transformers import ImageGPTModel
            processor = load_processor(model_name, processor_type="igptProcessor")
            model = ImageGPTModel.from_pretrained(model_name)
            return model, processor

        case "swin":
            from transformers import SwinForMaskedImageModeling
            processor = load_processor(model_name, processor_type="AutoProcessor")
            model = SwinForMaskedImageModeling.from_pretrained(model_name)
            return model, processor

        case "convnext":
            from transformers import ConvNextForImageClassification
            processor = load_processor(model_name, processor_type="convnextProcessor")
            model = ConvNextForImageClassification.from_pretrained(model_name)
            return model, processor

        case "resnet":
            from transformers import ResNetForImageClassification
            processor = load_processor(model_name, processor_type="AutoProcessor")
            model = ResNetForImageClassification.from_pretrained(model_name)
            return model, processor

        case "timm":
            import timm
            #load model
            model_name = model_name.split("/")[1]
            model = timm.create_model(model_name, pretrained=True)
            # get model specific transforms (normalization, resize)
            data_config = timm.data.resolve_model_data_config(model)
            transforms = timm.data.create_transform(**data_config, is_training=False)
            #return
            return model, transforms
            
        #multimodal 🖼️/📚 --- 
        case "clip":
            from transformers import CLIPModel
            model = CLIPModel.from_pretrained(model_name)
            processor = load_processor(model_name, processor_type="CLIP")
            return model, processor
            
        case _:
            raise ValueError(f"model_type not valid ")
         
    #END OF MATCH#
    
