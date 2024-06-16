from functools import partial

from activation_extractor.model_functions.model_types import model_types
from activation_extractor.model_functions.load_models import load_model
from activation_extractor.model_functions.tokenize_funs import define_tokenize_function
from activation_extractor.model_functions.inference_funs import define_inference_function

# Inferencer Class =====================
class Inferencer:
    """
    This class, when initialized with a model name
    - Loads model, tokenizer
    - Defines tokenization function
    - Defines inference function
    - Can be used to perform inference
    
    :param model_name: name of the model (i.e. as in huggingface)
    :type model_name: str
    :param data_type: allowed - "sequence", 
    :type data_type: str
    :param device: device (cpu, cuda...)
    :type device: str
    :param half: this variable decides if the pytorch model should be halfed
    :type device: bool
    
    :return: the corresponding hook function
    :rtype: function
    """
    def __init__(self, model_name, data_type="sequence", device="cpu", half=False):
        self.model_name = model_name
        self.device = device
        
        #model type
        for model_type in model_types[data_type]:
            if model_type in model_name:
                self.model_type=model_type
                break
            
        #tokenizer, model
        self.model, self.tokenizer = load_model(model_name, model_type=self.model_type)
        self.model.to(device)
        self.model.eval()
        if half: self.model.half()

        #tokenize, inference functions
        self.tokenize_fun = define_tokenize_function(self.model_type, self.tokenizer)
        self.inference_fun = define_inference_function(self.model_type, self.model, 
                                                       self.tokenizer, self.device)
        
    def tokenize(self, sequence_input):
        """
        Tokenize a sequence batch.
        
        :param sequence_input: list of sequences
        :type sequence_input: list of strings
        :return: the tokenized inputs
        """
        tokenized_inputs = self.tokenize_fun(sequence_input)
        return tokenized_inputs

    def inference(self, tokenized_input):
        """
        Do inference on a tokenized batch.
        
        :param tokenized_input: tokenized input sequences
        :return: the model outputs
        """
        outputs = self.inference_fun(tokenized_input, self.device)
        return outputs

   
