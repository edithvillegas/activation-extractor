"""
Defines a tokenizer wrapper function for the models included by default.
"""
import re

def define_tokenize_function(model_type, tokenizer, device=None):
    """
    Define the right function to tokenize the inputs based on the model type.
    This function is called inside inferencer.tokenizer().

    :param model_type: the model type (from the list in activation_extractor.model_functions.model_types)
    :type model_type: str
    :param tokenizer: the loaded tokenizer object

    :return: the function used to tokenize the inputs
    """
    match model_type:

        #🥩,🧬,🧬,🧬 
        case "esm" | "nucleotide-transformer" | "hyenadna" | "caduceus" | "dnabert":  
            #### start function definition
            def tokenize_fun(sequence_inputs, **kwargs):
                tokenized = tokenizer.batch_encode_plus(sequence_inputs, 
                                                             return_tensors="pt", 
                                                              padding="longest", 
                                                              **kwargs
                                                              )
                return tokenized
            #### end function definition

        #🧬
        case "evo":
            from evo.scoring import prepare_batch
            #### start function definition
            def tokenize_fun(sequence_inputs, **kwargs):
                tokens_ids, seq_lengths = prepare_batch(
                sequence_inputs,
                tokenizer,
                prepend_bos=False,
                device=device,
                )
                tokenized=dict()
                tokenized["input_ids"]=tokens_ids
                return tokenized
            #### end function definition

       
        #🥩
        case "prot_t5" | "prot_bert" | "prot_xlnet" | "prot_electra" : 
            #### start function definition
            def tokenize_fun(sequence_inputs, **kwargs):
                # replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
                sequence_inputs = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_inputs]
                tokenized = tokenizer.batch_encode_plus(sequence_inputs, add_special_tokens=True, 
                                                        padding="longest", return_tensors='pt')
                return tokenized 
             #### end function definition

        #🥩 protein sequence, ⛓️🥩 3di structural sequence
        case "prostt5" : 
            #### start function definition
            def tokenize_fun(sequence_inputs, sequence_type=None, **kwargs):
                # replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
                sequence_inputs = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_inputs]
                
                if sequence_type is None:
                    # add pre-fixes accordingly (this already expects 3Di-sequences to be lower-case)
                    # if you go from AAs to 3Di (or if you want to embed AAs), you need to prepend "<AA2fold>"
                    # if you go from 3Di to AAs (or if you want to embed 3Di), you need to prepend "<fold2AA>"
                    sequence_inputs = [ "<AA2fold>" + " " + s if s.isupper() else "<fold2AA>" + " " + s
                                          for s in sequence_inputs
                                        ]
                else:
                    #if sequence type is given, overwrite the above
                    sequence_inputs = [ "<AA2fold>" + " " + s if sequence_type=="protein" else "<fold2AA>" + " " + s
                                          for s in sequence_inputs
                                        ]
                
                
                tokenized = tokenizer.batch_encode_plus(sequence_inputs, add_special_tokens=True, 
                                                        padding="longest", return_tensors='pt')
                return tokenized 
             #### end function definition
    
        #🥩
        case "ankh":
            #### start function definition
            def tokenize_fun(sequence_inputs, **kwargs):
                sequence_inputs = [list(seq) for seq in sequence_inputs]
                tokenized = tokenizer.batch_encode_plus(sequence_inputs, 
                                    add_special_tokens=True, 
                                    padding="longest",
                                    is_split_into_words=True, 
                                    return_tensors="pt")
                return tokenized
           #### end function definition

    #return the rightly defined tokenizer function
    return tokenize_fun
