from functools import partial

from activation_extractor.model_functions.model_types import model_types
from activation_extractor.model_functions.load_models import load_model
from activation_extractor.model_functions.inference_funs import define_inference_function
from activation_extractor.model_functions.tokenize_funs import define_tokenize_function
from activation_extractor.model_functions.process_funs import define_process_function

import pandas as pd

# Inferencer Class =====================
class InferencerBase:
    """
    This class, when initialized with a model name:
    
    * Loads model, tokenizer
    * Defines tokenization function
    * Defines inference function
    * Can be used to perform inference
    
    :param model_name: name of the model (i.e. as in huggingface)
    :type model_name: str
    :param data_type: allowed - "sequence", 
    :type data_type: str
    :param device: device (cpu, cuda...)
    :type device: str
    :param half: this variable decides if the pytorch model should be halved
    :type device: bool
    
    :return: the corresponding hook function
    :rtype: function
    """
    def __init__(self, model_name, device="cpu", half=False):
        self.model_name = model_name
        self.device = device
        
        #model type
        self.model_type = model_types[model_types.model_name==model_name].model_type.tolist()[0]
        self.modality = model_types[model_types.model_name==model_name].modality.tolist()[0]
            
        #tokenizer, model
        self.model, self.processor = load_model(model_name, model_type=self.model_type)
        self.model.to(device)
        self.model.eval()
        if half: self.model.half()

        #define data pre-processing (tokenization, image processing, etc.)
        match self.modality:
            case "sequence" | "text":
                self.process_fun = define_tokenize_function(self.model_type, 
                                                            self.processor)
            case "image" | "image-text":
                self.process_fun = define_process_function(self.model_type, 
                                                            self.processor)

        #inference function
        self.inference_fun = define_inference_function(self.model_type, 
                                                               self.model, 
                                                               self.processor, 
                                                               self.device)
        
    def process(self, input_data, **kwargs):
        """
        Process input data (tokenize or image processing).
        The tokenizer works with batches of sequences (list of strings).
        The image processing so far with one image.
        
        :param input_data: list of sequences or image
        :type input_data: list of strings or image 
        :return: the processed inputs
        """
        processed_input = self.process_fun(input_data, **kwargs)
        return processed_input

    def inference(self, processed_input):
        """
        Do inference on processed inputs.
        
        :param processed_input: processed inputs
        :return: the model outputs
        """
        outputs = self.inference_fun(processed_input, self.device)
        return outputs

   
