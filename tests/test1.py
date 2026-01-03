import coregtor
from coregtor.pipeline import Pipeline,TargetResults,PipelineResults

# Additional imports
from pathlib import Path
import pandas as pd
import os
import json


config = {
  "target_gene": ["ZNF354C","NMRAL1","ZNF443","E2F8"], ### change this to run for just one exp
  "create_model": {
    "model": "rf",
    "model_options": {
      "max_depth": 5,
      "n_estimators": 1000,
      "n_jobs":5
    }
  },
  "tree_paths": {},
  "create_context": {
    "method": "tree_paths"
  },
  "transform_context": [
    {
      "id": "default",
      "method": "gene_frequency",
      "normalize": False,
      "min_frequency": 1
    }
  ],
  "compare_context": [
    {
      "id": "default",
      "method": "cosine",
      "transformation_id": "default",
      "convert_to_distance": False
    }
  ],
  "checkpointing": True,
  "force_fresh": False,
  "input": {
    "expression": "$HOME/projects/bio-datasets/bladder/gene_tpm_v10_bladder.gct",
    "tflist": "$HOME/projects/bio-datasets/bladder/tf20.csv"
  },
  "paths":{
      "temp":"$HOME/projects/temp-results"
  },
  "clustering":[
      {
          "id":"default",
          "matrix_id":"default",
          "method":"hierarchical_clustering",
          "method_options":{
              "auto_threshold":"inconsistency"
          }
      },
      {
          "id":"test1",
          "matrix_id":"default",
          "method":"hierarchical_clustering",
          "method_options":{
              "distance_threshold":0.75
          }
      }
  ],
  "result_generation":{
      "n_jobs":2,
      "rerun":True
  }
}

def get_a_path(pth):
    return Path(os.path.expanduser(os.path.expandvars(pth))) 
    




def run_pipeline():
  expression_data = coregtor.utils.exp.read_GE_data(get_a_path(config["input"]["expression"]))
  tflist = pd.read_csv(get_a_path(config["input"]["tflist"]),names=["gene_name"], header=None)
  pipeline = Pipeline(expression_data,tflist,config)
  pipeline.run()
  # details = pipeline.run_details()

  # if config["checkpointing"]:
  #     output_path = get_a_path(config["output_dir"]) / f"{pipeline.title}.json"
  #     with open(output_path, "w") as f:
  #         json.dump(details, f, indent=1)
  # else:
  #     print(details)

def run_single_target(target: str):
        expression_data = coregtor.utils.exp.read_GE_data(get_a_path(config["input"]["expression"]))
        tflist = pd.read_csv(get_a_path(config["input"]["tflist"]),names=["gene_name"], header=None)
        
        if target not in expression_data.columns:
            raise ValueError(f"Target '{target}' not in expression data")
        
        # 2. Model input + training
        X, Y = coregtor.create_model_input(expression_data, target, tflist)
        # 3. Model training
        model_config = config.get("create_model")
        model = coregtor.create_model(X, Y, **model_config)
        #print(model)
        # 4. Tree paths
        tree_paths_config = config.get("tree_paths")
        paths = coregtor.tree_paths(model, X, Y, **tree_paths_config)
    
        # 5. Create contexts
        create_context_config = config.get("create_context")

        print(paths)
        contexts = coregtor.create_context(paths, **create_context_config)
        print(contexts)
        # 6. Transform contexts
        transform_configs = config.get("transform_context", [])
        transform_results = []
        
        for t_config in transform_configs:
            transformed = coregtor.transform_context(contexts, **t_config)
            transform_results.append({"id": t_config["id"], "result": transformed})
        
        # 7. Compare contexts
        compare_configs = config.get("compare_context", [])
        comparison_results = []
        
        for c_config in compare_configs:
            transform_found = next((d for d in transform_results if d.get("id") == c_config.get("transformation_id")), None)
            if transform_found is None:
                continue
            transformed_data = transform_found["result"]
            print(transformed_data)
            matrix = coregtor.compare_context(transformed_data, **c_config)    
            comparison_results.append({"id": c_config["id"], "result": matrix})

   
# run_single_target("ESX1")

# run_pipeline()

# print(expression_data[ expression_data["ESX1"] >0 ]["ESX1"]  )

def target(et,t):
    tr = TargetResults(config,exp_title=et,target=t)
    tr.generate_result_files(rerun=False)
    st = tr.generate_result(rerun=True)
    print(st)
    # stats = tr.get_stats()
    # print(stats)
    # fpaths = tr.generate_figures()
    # print(fpaths)
    # print(tr)

def exp_res(et):
    expression_data = coregtor.utils.exp.read_GE_data(get_a_path(config["input"]["expression"]))
    tflist = pd.read_csv(get_a_path(config["input"]["tflist"]),names=["gene_name"], header=None)
    resp = PipelineResults(options=config,exp_title=et,tflist=tflist)
    resp.run() 
    resp.generate_full_exp_results()


# target("Exp_1767377217","ZNF354C")
# target("Exp_1767018345","NMRAL1")

exp_res("Exp_1767377217")