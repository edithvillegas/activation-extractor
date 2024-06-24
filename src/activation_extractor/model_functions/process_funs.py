"""
Defines a processor wrapper function for the models included by default.
"""
import re

def define_process_function(model_type, processor, device=None):
    """
    Define the right function to process the inputs based on the model type.
    This function is called inside inferencer.processor().

    :param model_type: the model type (from the list in activation_extractor.model_functions.model_types)
    :type model_type: str
    :param processor: the loaded tokenizer object

    :return: the function used to process the inputs
    """
    match model_type:
        #images 🖼️
        case "vit":
            #### start function definition
            def process_fun(image, **kwargs):
                processed = processor(images=image, return_tensors="pt")
                return processed
            #### end function definition
    
        #multimodal 🖼️/📚 
        case "clip" :  
            #### start function definition
            def process_fun(inputs, **kwargs):
                #This function takes as input a dictionary with text and image keys
                processed = processor(text=inputs["text"], images=inputs["image"], 
                                      return_tensors="pt", padding=True)
                return processed
            #### end function definition

        case _:
            raise ValueError(f"model_type not valid ")
         
    #return the rightly defined tokenizer function
    return process_fun