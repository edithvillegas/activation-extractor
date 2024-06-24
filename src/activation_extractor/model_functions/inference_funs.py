"""
This file defines an inferencer wrapper for the included models.
"""
import torch

def define_inference_function(model_type, model, tokenizer, device):
    """
    Define the right function to do inference based on the model type.
    The resulting function is called as inferencer.inference().
    The functions move the tokenized input to ``device`` before performing inference.

    :param model_type: the model type (from the list in activation_extractor.model_functions.model_types)
    :type model_type: str
    :param model_type: the loaded pytorch model
    :param tokenizer: the loaded tokenizer object
    :param device: the device (cpu, cuda...)
    :type device: str

    :return: the function used to do the inference
    """
    match model_type:

        #🥩,🧬 
        case "esm" | "nucleotide-transformer" :  
            #### start function definition
            def inference_fun(tokenized_inputs, device, **kwargs):

                tokens_ids = tokenized_inputs["input_ids"].to(device)
                attention_mask = tokens_ids != tokenizer.pad_token_id

                #set default parameters
                if 'output_hidden_states' in kwargs: 
                    output_hidden_states=kwargs['output_hidden_states']
                else:
                    output_hidden_states=False

                #inference
                outputs = model(
                    tokens_ids,
                    attention_mask=attention_mask,
                    encoder_attention_mask=attention_mask,
                    output_hidden_states=output_hidden_states
                )
                
                return outputs
            #### end function definition

        #🥩, 🥩
        case "prot_t5" | "ankh" :
            #### start function definition
            def inference_fun(tokenized_inputs, device, **kwargs):
                tokens_ids = torch.tensor(tokenized_inputs['input_ids']).to(device)
                attention_mask = torch.tensor(tokenized_inputs['attention_mask']).to(device)
                
                outputs = model(input_ids=tokens_ids, 
                                attention_mask=attention_mask)
                return outputs
            #### end function definition    

        #🥩
        case "prot_bert" | "prot_xlnet" | "prot_electra":
            #### start function definition
            def inference_fun(tokenized_inputs, device, **kwargs):
                
                for key in tokenized_inputs.keys():
                    tokenized_inputs[key]=tokenized_inputs[key].to(device)
                    
                outputs = model(**tokenized_inputs)
                return outputs
            #### end function definition   

        case "prostt5":
            #### start function definition
            def inference_fun(tokenized_inputs, device, **kwargs):
                
                for key in tokenized_inputs.keys():
                    tokenized_inputs[key]=tokenized_inputs[key].to(device)
                    
                outputs = model(tokenized_inputs["input_ids"],
                               attention_mask=tokenized_inputs["attention_mask"])
                return outputs
            #### end function definition   
            

        #default inference
        #🧬, 🧬
        case "hyenadna" | "evo" | "caduceus" :
            #### start function definition
            def inference_fun(tokenized_inputs, device, **kwargs):
                tokens_ids = tokenized_inputs["input_ids"].to(device)
                outputs = model(tokens_ids)
                return outputs
            #### end function definition    

    #return rightly defined inference function
    return inference_fun
    
