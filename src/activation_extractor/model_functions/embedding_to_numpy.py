import torch
from transformers.modeling_outputs import (MaskedLMOutput, 
                                            BaseModelOutputWithNoAttention, 
                                            BaseModelOutputWithPastAndCrossAttentions)

def embedding_to_numpy(embeddings):
    """
    Converts different types of module outputs to a numpy array.
    Handles different cases for the different models.
    Additionally, moves from GPU to CPU. 

    :param embedding: Intermediate output object from a pytorch model layer/module.
    
    :return: intermediate output as a numpy array
    :rtype: numpy array
    """
    
    #convert list to array (for protT5 tokens)
    if isinstance(embeddings, list): 
        embeddings = np.array(embeddings) 
        return embeddings
    
    #convert output object to tensor -------------------------
    #intermediate activations
    if isinstance(embeddings, tuple): 
        embeddings = embeddings[0] 
        
    #NT/ESM output object
    if isinstance(embeddings, MaskedLMOutput): 
        embeddings = embeddings['logits'] 
        
    #HYENA or ANKH output object
    if (isinstance(embeddings, BaseModelOutputWithNoAttention) 
        or isinstance(embeddings, BaseModelOutputWithPastAndCrossAttentions)):
        embeddings = embeddings['last_hidden_state']  
    #convert output object to tensor -------------------------
    
    #float conversions for tensors 
    if torch.is_tensor(embeddings):
        if embeddings.dtype==torch.bfloat16: embeddings = embeddings.float() 
   
    #to CPU numpy array
    embeddings = embeddings.cpu().detach().numpy()   
    
    return embeddings
