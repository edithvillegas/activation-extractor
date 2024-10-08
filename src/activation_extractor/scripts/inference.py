#imports =================================================================
import sys, os

import numpy as np
import pandas as pd
import shutil, time
from datetime import datetime
import argparse
import pickle

import torch
from torch.cuda import memory_allocated, empty_cache
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset

from activation_extractor import Inferencer, IntermediateExtractor, get_layers_to_hook
from activation_extractor.model_functions.embedding_to_numpy import embedding_to_numpy
#import all collate functions from:
from activation_extractor.scripts.collate_functions import *

# Parsing Arguments -----------------------------------------------------------------------
def argument_parser(): 
    parser = argparse.ArgumentParser()

    #base arguments 
    parser.add_argument('--model_name', type=str) #model name
    parser.add_argument('--batch_size', type=int) #model batch size for inference
    parser.add_argument('--data_type', type=str) #dna, protein, image, multimodal
    parser.add_argument('--modality', type=str) #sequence, image-text

    #output
    parser.add_argument('--output_folder', type=str) #save representations here
    parser.add_argument('--emb_format', type=str, default='mean') #mean, LT (last token)
    parser.add_argument('--save_mean', type=int, default=0)
    parser.add_argument('--save_lt', type=int, default=0)
    parser.add_argument('--save_ft', type=int, default=0)
    parser.add_argument('--save_method', type=str, default='numpy') #numpy or numpy_compressed
    parser.add_argument('--sequence_axis', type=int, default=1) #sequence length axis
    
    #data source
    parser.add_argument('--data_source', type=str, default='local') #huggingface or local
    parser.add_argument('--target_col', type=str, default='sequence') #column with the data for inference
    # if dataset from huggingface
    parser.add_argument('--dataset_name', type=str) #huggingface dataset name
    parser.add_argument('--dataset_partition', type=str) #huggingface dataset partition
    # if dataset is local
    parser.add_argument('--input_path', type=str) #csv file path

    #optional arguments
    parser.add_argument('--max_length', type=int, default=None) #default maximum sequence length
    parser.add_argument('--max_batches', type=int, default=None) #break when reach max batches
    parser.add_argument('--min_batches', type=int, default=None) #break when reach max batches
    #multimodal image/text dataset
    parser.add_argument('--image_source', type=str, default="local") #download images or get them from local folder
    parser.add_argument('--image_dir', type=str, default=None) #download images or get them from local folder
    #multimodal protein sequence/structure
    parser.add_argument('--sequence_inf', type=int, default=1)
    parser.add_argument('--structure_inf', type=int, default=1)

    #parse arguments
    args = parser.parse_args()
    
    #assign variables 
    model_name = args.model_name
    output_folder = args.output_folder

    #optional arguments
    max_batches = args.max_batches
    min_batches = args.min_batches

    #saving arguments
    save_args = {
            "save_method":args.save_method,
            "sequence_axis": args.sequence_axis,
    }
    if args.emb_format=="list":
        save_args["emb_formats"]=[]
        if args.save_mean:
            save_args["emb_formats"].append("mean")
        if args.save_lt:
            save_args["emb_formats"].append("LT")
        if args.save_ft:
            save_args["emb_formats"].append("FT")
    else:
        save_args["emb_formats"]=[args.emb_format]

    #data arguments
    data_args = {
            "data_type":args.data_type,
            "data_source":args.data_source,
            "modality":args.modality,
            "target_col":args.target_col,
            "batch_size":args.batch_size,
            #ms coco
            "image_dir":args.image_dir,
            "image_source": args.image_source,
        }

    if args.data_source=="local":
        data_args["input_path"] = args.input_path
        
    elif args.data_source=="huggingface":
        data_args["dataset_name"] = args.dataset_name
        data_args["dataset_partition"] = args.dataset_partition

    if args.data_type in ["dna", "protein"]:
        data_args["max_length"] = args.max_length

    #multimodality optional arguments
    inference_args = dict()
    if args.sequence_inf==0:
        inference_args['sequence']=False
    if args.structure_inf==0:
        inference_args['structure']=False

    return (model_name, output_folder, save_args, max_batches, min_batches, data_args, inference_args)

# Loading a dataset -----------------------------------------------------------------
def load_the_data(
                #general data info
                data_type, data_source, target_col=None, 
                modality=None,
                #optional huggingface info
                dataset_name=None, dataset_partition=None, 
                #optional local dataset info
                input_path=None,       
                #data loader options
                batch_size=8, collate_fn=None,
                #data type options
                max_length=None,
                #ms coco
                image_dir=None,image_source=None,
                 ):

    #load dataset from source 
    if data_source=="huggingface":
        try:
            dataset = load_dataset(dataset_name,
                          trust_remote_code=True)
        except TypeError:
            dataset = load_dataset(dataset_name)
            
        dataset = dataset[dataset_partition]
        
    elif data_source=="local":
        dataset = pd.read_csv(input_path, index_col=False, comment="#")

    #get target column
    if data_type in ["dna", "protein", "protein-str"]:
        dataset = dataset[target_col]

    #get default collate function based on data type
    if collate_fn is None:
        
        if data_type in ["protein", "dna"]:
            #truncate sequence to max length
            collate_fn = create_collate_to_max_length(max_length)
            
            if data_type=="protein":
                #make sure the letters are all caps
                collate_fn = compose_collate_fns(collate_fn, collate_to_upper)

        if data_type=="mscoco":
            #MS COCO!!
            #deduplicate picture descriptions
            dataset = pd.DataFrame(dataset)
            dataset=dataset.drop_duplicates(subset=['URL'])
            dataset = Dataset.from_pandas(dataset)
            
            #collate function
            collate_fn = create_collate_mscoco(image_source,
                          modality=modality,
                          image_dir=image_dir,
                         )

        if data_type=="protein-str":
            collate_fn = pdb_path_collate
    #create data loader
    data_loader = DataLoader(dataset, batch_size=batch_size, 
                             shuffle=False, collate_fn=collate_fn)

    return data_loader


# SCRIPT ================================================================================
def main_inference(model_name, output_folder, save_args, max_batches, min_batches, data_args, inference_args):    
    #make folders
    output_folder=f"{output_folder}/{model_name}"
    folders = [
        output_folder,
        f"{output_folder}/tokens",
        f"{output_folder}/outputs",
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        os.chmod(folder, mode=0o777)

    #print start to log file
    logfile_path = f"{output_folder}/inference.log"
    logfile = open(logfile_path, "a")
    print(f'====================', file=logfile, flush=True)
    print(f'✔️ Script started on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
          file=logfile, flush=True)

    #device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #load data
    data_loader = load_the_data(**data_args)

    #load model
    inferencer = Inferencer(model_name, device=device, half=False)

    # print to log
    print(f'✔️ {model_name} loaded', file=logfile, flush=True)
    print(f'Output folder is: {output_folder}')
    print(f"✔️ {model_name}", file=sys.stderr)

    #intermediate activation extractor
    layers_to_hook, structure = get_layers_to_hook(model=inferencer.model, model_type=inferencer.model_type, 
                                                   modality=data_args["modality"], return_structure=True
                                                  )
    extractor = IntermediateExtractor(inferencer.model, layers_to_hook)
    extractor.register_hooks()

    #save inference params to file
    with open(f'{output_folder}/hooked_layers.pkl', 'wb') as file:
        pickle.dump(structure, file)
        
    with open(f"{output_folder}/inference_args.txt", 'w') as file:
        file.write("🔸 Inference Arguments \n")
        file.write(f"Model name: {model_name} \n") 
        file.write("🔸 Save Arguments \n")
        file.write(str(save_args)+"\n") 
        file.write("🔸 Data Arguments \n")
        file.write(str(data_args)+"\n")
        file.write("🔸 Hooked Layers \n")
        file.write(str(structure))

    #Inference Loop ------------------------------------------
    #measure total execution time
    start_total_time = time.time()

    for batch_i, batch in enumerate(data_loader):  
        #skip batches
        if min_batches is not None:
            if batch_i<min_batches:
                pass
                
        ### Inference Part ###
        #process
        processed = inferencer.process(batch)
        
        #inference
        try:
            start_inference_time = time.time()
            outputs = inferencer.inference(processed, **inference_args)
            inference_time = time.time() - start_inference_time
        except Exception as e:
            print(f"❌ {model_name}", file=logfile, flush=True)
            print(f'Exception: {e}', file=logfile, flush=True)
            print(f'Exception: {e}', file=sys.stderr)
            #exit
            logfile.close()
            sys.exit()

        ### Saving Part ###
        ## intermediate activations
        extractor.save_outputs(output_folder=f"{output_folder}",
                               output_id=str(batch_i),
                               move_to_cpu=True, 
                               **save_args) #also creates folder

        
        if data_args["data_type"] in ["dna", "protein", "text"]:
            #tokens
            tokens=embedding_to_numpy(processed['input_ids'])
            np.save(f'{output_folder}/{tokens}/{batch_i}/tokens_ids.npy', tokens)
            
            ## outputs
            outputs = embedding_to_numpy(outputs)
            np.save(f'{output_folder}/{outputs}/{batch_i}/outputs.npy', outputs)
        
        print(f"Completed batch {batch_i} in {inference_time:.4f} s", file=logfile, flush=True)

        #get out of loop if reached max batches
        if max_batches is not None:
            if batch_i==max_batches:
                break

        #clear memory
        del outputs
        empty_cache()
    #loop over batches ---------------------------
    
    #total execution time
    total_time = time.time() - start_total_time
    print(f'Total time: {total_time}', file=logfile, flush=True)
    
    #Exit --------------------------
    #close log file
    print(f'✔️ Script finished on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
          file=logfile, flush=True)
    logfile.close()

def main():
    #parse arguments
    model_name, output_folder, save_args, max_batches, min_batches, data_args, inference_args = argument_parser()
    
    #execute main
    main_inference(model_name, output_folder, save_args, max_batches, min_batches, data_args, inference_args)
    
#Execute main function
if __name__ == "__main__":
    main()
