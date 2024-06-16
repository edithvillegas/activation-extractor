from activation_extractor.model_functions.model_types import model_types
from activation_extractor.model_functions.load_models import load_model
from activation_extractor.model_functions.tokenize_funs import define_tokenize_function
from activation_extractor.model_functions.inference_funs import define_inference_function

# Inferencer Class =====================
class Inferencer:
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

        #tokenizer, inference functions
        self.tokenize = define_tokenize_function(self.model_type, self.tokenizer)
        self.inference = define_inference_function(self.model_type, self.model, self.device)

   
