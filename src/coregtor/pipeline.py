import json
import time
import joblib
import jsonschema
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, List, Tuple
from contextlib import contextmanager
import hashlib
from sklearn.metrics import r2_score
import psutil
import os
import numpy as np 
import time

from joblib import Parallel, delayed

# Core CoRegTor imports
from coregtor.forest import create_model_input, create_model, tree_paths
from coregtor.context import create_context, transform_context, compare_context
from coregtor.utils.error import CoRegTorError
from coregtor.clusters import identify_coregulators
# Final ID-Driven Schema
SCHEMA = {
    "type": "object",
    "required": [],
    "properties": {
        "target_genes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of target genes. [] = auto-discover TFs in data."
        },
        # === FUNCTION-SPECIFIC CONFIGS ===
        "create_model": {
            "type": "object",
            "properties": {
                "model": {"type": "string", "enum": ["rf", "et"],"default":"rf"},
                "model_options": {
                    "type": "object", 
                    "description": "sklearn.ensemble params",
                    "default":{"max_depth": 5, "n_estimators": 1000}
                }
            },
            "description": "create_model() configuration"
        },
        "tree_paths": {
            "type": "object",
            "default": {},
            "description": "tree_paths() parameters (future expansion)"
        },
        "create_context": {
            "type": "object",
            "properties": {
                "method": {"type": "string","default":"tree_paths"}
            },
            "description": "Generate context"
        },
        "transform_context": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Unique identifier",
                        "default": "default"
                    },
                    "method": {
                        "type": "string",
                        "description": "transform_context method",
                        "default": "gene_frequency",
                    },
                    "normalize": {
                        "type": "boolean",
                        "description": "for gene_frequency method, If True, normalize frequencies to proportions (default: False) ",
                        "default": False
                    },
                    "min_frequency": {
                        "type": "integer",
                        "description": "for gene_frequency method, Minimum frequency threshold to include gene (default: 1) ",
                        "default": 1
                    },
                },
                "required": ["id", "method"],
                "additionalProperties": True
            },
            "description": "Context transformation configs (in an array)"
        },
        "compare_context": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string", 
                        "description": "Unique identifier",
                        "default":"default"
                      },
                    "method": {
                        "type": "string", 
                        "description": "Similarity/distance metric name",
                        "default":"cosine"
                      },
                    "transformation_id":{
                        "type": "string", 
                        "description": "The Id of the transformed_context to be used",
                        "default":"default"
                    },
                    "convert_to_distance": {
                        "type": "boolean", 
                        "default": False,
                        "description":"Convert similarity to distance (1 - similarity)"
                    }
                },
                "required": ["id", "method","transformation_id"],
                "additionalProperties": True
            },
            "description": "Context Comparison configs"
        },
        # === PIPELINE CONTROL ===
        "checkpointing": {"type": "boolean", "default": True},
        "force_fresh": {"type": "boolean", "default": False},
        "input":{"type":"object","default":{"expression":"","tflist":""}},
        "paths":{"input":{"type":"str","default":""},"output":{"type":"str","default":""},"temp":{"type":"str","default":""}},

        ### Clustering and result generation
        "clustering": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string", 
                        "description": "Unique identifier for this clustering method",
                        "default":"default"
                      },
                    "matrix_id":{
                        "type": "string", 
                        "description": "The Id of the transformed_context matrix to be used",
                        "default":"default"
                    },
                    "method": {
                        "type": "string",
                        "enum": ["hierarchical_clustering"],
                        "default": "hierarchical_clustering"
                    },
                    "method_options": {
                        "type": "object",
                        "default":{"auto_threshold":'inconsistency'}
                    }
                },
                "required": ["id", "matrix_id"],
                "additionalProperties": True
            },
            "description": "Clustering configurations."
        },
        "result_generation":{
            "type":"object",
            "properties":{
                "n_jobs":{
                    "type:":"integer",
                    "default":1,
                    "description":"How many targets for run in parallel for generating individual results"
                },
                "rerun":{
                    "type":"boolean",
                    "default":False,
                    "description":"Will rerun the result generation pipeline"
                },
                "generate_figures":{
                    "type":"boolean",
                    "default":True,
                    "description":"If true, generates figures in a temp folder"
                }
            },
            "additionalProperties": True
        }
    }
}


class Pipeline:
    def __init__(self,expression_data:pd.DataFrame,tflist:list,options: Dict[str, Any],exp_title: str = None):
        
        self.expression_data = expression_data
        
        if tflist is None or len(tflist)==0:
            raise ValueError("TF list not provided")
        self.tflist = tflist
        
        default_options = Pipeline._generate_default_config_dict()
        input_options = options
        options = {**default_options,**(input_options or {})}
        jsonschema.validate(instance=options, schema=SCHEMA)
        self.options = options
        
        self.results: Dict[str, Dict[str, Any]] = {}
        self.configs: Dict[str, Dict[str, Any]] = {}
        self.stats: Dict[str, Dict[str, Any]] = {}
        self.status: Dict[str, Dict[str, Any]] = {}
        
        self.title = exp_title.replace(" ", "_") if exp_title else f"Exp_{int(time.time())}"

        if self.options.get("checkpointing", True):
            
            self.checkpoint_dir = Path(os.path.expanduser(os.path.expandvars(self.options.get("paths").get("temp")))) / self.title 
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    

    ### manage pipeline checkpoints 
    def _get_pipeline_hash(self) -> str:
        return hashlib.md5(json.dumps(self.options, sort_keys=True).encode()).hexdigest()
    
    def _checkpoint_file(self, target: str) -> Path:
        return self.checkpoint_dir / f"{target}.pkl"
        
    def _checkpoint_exists(self, target: str) -> bool:
        if not self.options.get("checkpointing", True) or self.options.get("force_fresh", False):
            return False
        checkpoint_file = self._checkpoint_file(target)
        if not checkpoint_file.exists():
            return False
        try:
            checkpoint = joblib.load(checkpoint_file)
            return checkpoint.get("pipeline_hash") == self._get_pipeline_hash()
        except:
            return False
    
    def _save_checkpoint(self, target: str):
        checkpoint = {
            "pipeline_hash": self._get_pipeline_hash(),
            "timestamp": time.time(),
            "results": self.results[target],
            "stats": self.stats[target],
            "success": self.status[target]
        }
        joblib.dump(checkpoint, self._checkpoint_file(target))

        # if self.status[target]["success"]:
        #     print("saving the similarity matrix file separately")
        #     for item in self.results[target]["comparison_results"]:
        #         file_name = f"{target}_sim_{item["id"]}.csv"
        #         item["result"].to_csv(self.checkpoint_dir/file_name)

        
        ## To make the pipeline more efficient, clearing data after success saving of results 
        self.results[target] = None
        self.stats[target] = None
        self.status[target] = None

    
    def _load_checkpoint(self, target: str):
        # to save space, skip loading already finished experiments
        load_old_results = False
        if load_old_results : 
          checkpoint = joblib.load(self._checkpoint_file(target))
          #print(f"Loaded checkpoint for {target}")
          self.results[target] = checkpoint["results"]
          self.stats[target] = checkpoint["stats"]

    
    def _auto_discover_targets(self) -> List[str]:
        tfs = set( self.tflist)
        genes_in_data = set(self.expression_data.columns)
        return sorted(list(tfs & genes_in_data))
    
    def _get_targets(self) -> List[str]:
        targets = self.options.get("target_genes", [])
        if targets is None or len(targets)==0:
            return self._auto_discover_targets()
        return targets
    
    def _run_single_target(self, target: str):
        print(f"Processing target: {target}")
        stats = {"timing": {}, "quality": {}}
        results = {}
        status = {"success":False, "error":""}
        
        try:
          # 1. Validate target exists
          if target not in self.expression_data.columns:
              raise ValueError(f"Target '{target}' not in expression data")
          
          # 2. Model input + training
          start_time = time.perf_counter()
          X, Y = create_model_input(self.expression_data, target, self.tflist)
          end_time = time.perf_counter()
          timing = end_time - start_time
          #print(f"model_input: {timing:.3f}s")
          stats["timing"]["model_input"] = timing
          
          # 3. Model training
          model_config = self.options.get("create_model")
          start_time = time.perf_counter()
          model = create_model(X, Y, **model_config)
          end_time = time.perf_counter()
          timing = end_time - start_time
          #print(f"model_train: {timing:.3f}s")
          stats["timing"]["model_train"] = timing
          
          results["model"] = model
          
          # 4. Tree paths
          tree_paths_config = self.options.get("tree_paths")
          start_time = time.perf_counter()
          paths = tree_paths(model, X, Y, **tree_paths_config)
          end_time = time.perf_counter()
          timing = end_time - start_time
          #print(f"paths_extract: {timing:.3f}s")
          stats["timing"]["paths_extract"] = timing
          stats["quality"]["n_paths"] = len(paths)
          stats["quality"]["n_unique_roots"] = paths["source"].nunique()
          results["paths"] = paths
          
          # 5. Create contexts
          create_context_config = self.options.get("create_context")
          start_time = time.perf_counter()
          contexts = create_context(paths, **create_context_config)
          end_time = time.perf_counter()
          timing = end_time - start_time
          #print(f"context_create: {timing:.3f}s")
          stats["timing"]["context_create"] = timing
          stats["quality"]["n_contexts"] = len(contexts)
          results["contexts"] = contexts
          
          # 6. Transform contexts
          transform_configs = self.options.get("transform_context", [])
          transform_results = []
          transform_stats = []
          
          for t_config in transform_configs:
              start_time = time.perf_counter()
              transformed = transform_context(contexts, **t_config)
              end_time = time.perf_counter()
              timing = end_time - start_time
              #print(f"transform_{t_config['id']}: {timing:.3f}s ")
              t_stat = {
                  "id": t_config["id"],
                  "n_genes": transformed.shape[1],
                  "n_rows": transformed.shape[0],
                  "time_s": timing
              }
              transform_results.append({"id": t_config["id"], "result": transformed})
              transform_stats.append(t_stat)
          
          results["transform_results"] = transform_results
          stats["transform_stats"] = transform_stats
          
          # 7. Compare contexts
          compare_configs = self.options.get("compare_context", [])
          comparison_results = []
          comparison_stats = []
          
          for c_config in compare_configs:
              transform_found = next((d for d in transform_results if d.get("id") == c_config.get("transformation_id")), None)
              if transform_found is None:
                  continue
              
              transformed_data = transform_found["result"]
              
              start_time = time.perf_counter()
              matrix = compare_context(transformed_data, **c_config)
              end_time = time.perf_counter()
              timing = end_time - start_time
              #print(f"compare_{c_config['id']}: {timing:.3f}s")
              
              c_stat = {
                  "id": c_config["id"],
                  "transformation_id": c_config["transformation_id"],
                  "time_s": timing
              }
              comparison_results.append({"id": c_config["id"], "result": matrix})
              comparison_stats.append(c_stat)
          
          results["comparison_results"] = comparison_results
          stats["comparison_stats"] = comparison_stats
          
          # Final stats
          stats["timing"]["total"] = sum(stats["timing"].values())
          stats["quality"]["n_transforms"] = len(transform_configs)
          stats["quality"]["n_comparisons"] = len(comparison_results)
          

          status["success"]= True
          status["error"] = ""

          self.results[target] = results
          self.stats[target] = stats
          self.status[target] = status
          
          if self.options.get("checkpointing", True):
              self._save_checkpoint(target)

        except CoRegTorError as e :
          print(e)
          status["success"]= False
          status["error"] = str(e) 
          if hasattr(e, 'code'):
              status["error_code"] = e.code  # Bonus: include code
          if hasattr(e, 'details'):
              status["error_details"] = e.details

          self.results[target] = None
          self.stats[target] = None
          self.status[target] = status
          
          if self.options.get("checkpointing", True):
              self._save_checkpoint(target)
    
    def run(self):
        targets = self._get_targets()
        for target in targets:
            if self._checkpoint_exists(target):
                # self._load_checkpoint(target)
                print(f"{target} already processed")
            else:
                self._run_single_target(target)
        print("Run success")
    

    @staticmethod
    def _generate_default_config_dict() -> Dict[str, Any]:
        """Generate default config dict from SCHEMA (preserves order)."""
        
        def extract_schema_defaults(node: Dict[str, Any]) -> Dict[str, Any]:
            result = {}
            
            if node.get("type") == "object" and "properties" in node:
                for prop_name in node["properties"]:
                    prop_schema = node["properties"][prop_name]
                    result[prop_name] = extract_schema_defaults(prop_schema)
                return result
            
            if node.get("type") == "array" and "items" in node:
                item_default = extract_schema_defaults(node["items"])
                return [item_default] if item_default else []
            
            if "default" in node:
                return node["default"]
            
            type_defaults = {
                "string": "", "boolean": False, "integer": 0, "number": 0.0,
                "array": [], "object": {}
            }
            return type_defaults.get(node.get("type"), {})
        
        default_config = extract_schema_defaults(SCHEMA)
        default_config["target_genes"] = []
        
        return default_config
    

class PipelineResults:
    def __init__(self,options: Dict[str, Any],tflist:list,exp_title: str = None):
    
        self.tflist = tflist
        default_options = Pipeline._generate_default_config_dict()
        input_options = options
        options = {**default_options,**(input_options or {})}
        jsonschema.validate(instance=options, schema=SCHEMA)
        self.options = options
        
        self.title = exp_title
            
        self.checkpoint_dir = Path(os.path.expanduser(os.path.expandvars(self.options.get("paths").get("temp")))) / self.title 
            
        
    def _auto_discover_targets(self) -> List[str]:
        tfs = set(self.tflist)
        genes_in_data = set(self.expression_data.columns)
        all_non_tf_targets = genes_in_data - tfs
        return sorted(list(all_non_tf_targets))
    
    def _get_targets(self) -> List[str]:
        targets = self.options.get("target_genes", [])
        if not targets:
            return self._auto_discover_targets()
        return targets

    def generate_full_exp_results(self):
        """Generate cluster_groups.csv first, then clean clusters.csv and cluster_configs.csv."""
        if not self.results:
            raise ValueError("Run generate_result first to populate self.results")
        
        full_clusters = pd.DataFrame()  # Single DF for everything
        config_rows = []
        
        for target, result in self.results.items():
            result_paths = result.get("result_paths", [])
            config_stats = result.get("config_stats", [])
            
            for path_str in result_paths:
                path = Path(path_str)
                if path.exists():
                    df = pd.read_csv(path)
                    df["target"] = target
                    full_clusters = pd.concat([full_clusters, df], ignore_index=True)
            
            for stat in config_stats:
                stat_copy = stat.copy()
                stat_copy["target"] = target
                stat_copy["exp_title"] = self.title
                config_rows.append(stat_copy)
        
        if full_clusters.empty:
            print("No cluster data found")
            return
            
        # Step 1: cluster_groups.csv grouped by clusteruid ONLY
        group_cols = ["cluster_uid"]  # Just cluster_uid
        grouped = full_clusters.groupby(group_cols).agg({
            "gene": lambda x: ";".join(sorted(set(x))),  # Semicolon-separated unique genes
            "score": "mean",
            "cluster_id": "first",     # First cluster_id
            "target": "first",         # First target
            "methodology": "first",    # First methodology
            "exp_details": "first"     # First exp_details
        }).reset_index()
        grouped  = grouped.drop(columns=["cluster_id"])
        grouped = grouped.sort_values(by=["target"])
        groups_path = self.checkpoint_dir / "cluster_groups.csv"
        grouped.to_csv(groups_path, index=False)
        print(f"Saved cluster_groups.csv: {len(grouped)} unique clusters")
         
        # Step 2: Clean clusters.csv (single drop operation)
        clean_clusters = full_clusters.drop(columns=["methodology", "exp_details"], errors="ignore")
        clusters_path = self.checkpoint_dir / "clusters.csv"
        clean_clusters.to_csv(clusters_path, index=False)
        print(f"Saved clusters.csv: {len(clean_clusters)} rows (cleaned)")
        
        # Step 3: cluster_configs.csv
        if config_rows:
            configs_df = pd.DataFrame(config_rows)
            configs_path = self.checkpoint_dir / "cluster_configs.csv"
            configs_df.to_csv(configs_path, index=False)
            print(f"Saved cluster_configs.csv: {len(config_rows)} rows")
        
        print("Full experiment results generated")




    def run(self):
        targets = self._get_targets()
        n_jobs = self.options.get("result_generation", {}).get("n_jobs", 1)
        rerun = self.options.get("result_generation", {}).get("rerun", False)
        
        self.results = {}  # Initialize class attribute for storage
        
        def process_target(target):
            tr = TargetResults(self.options, exp_title=self.title, target=target)
            tr.generate_result_files(rerun)
            result_dict = tr.generate_result(rerun)  # Returns dict
            print(f"Processed {target}")
            return target, result_dict  # Return target + dict for mapping
        
        target_results = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(process_target)(target) for target in targets
        )
        
        # Populate self.results: {target: result_dict}
        for target, result_dict in target_results:
            self.results[target] = result_dict

    

class TargetResults:
    def __init__(self,options: Dict[str, Any],exp_title,target:str): 
        default_options = Pipeline._generate_default_config_dict()
        input_options = options
        options = {**default_options,**(input_options or {})}
        jsonschema.validate(instance=options, schema=SCHEMA)
        self.options = options
        self.target = target
        self.exp_title = exp_title
        self.similarity = {} 
        self.stats = {}
        self.checkpoint_file = None
        self.load()
        
    def load(self):
        """Load target checkpoint file and validate existence."""

        if self.exp_title is None:
            raise ValueError("Exp title not provided")

        if self.target is None:
            raise ValueError("No target provided")
        
        self.checkpoint_dir = Path(os.path.expandvars(self.options["paths"]["temp"]))/ self.exp_title

        checkpoint_file1 =  self.checkpoint_dir / f"{self.target}.pkl"

        if not checkpoint_file1.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file1}")
        
        try:
            self.checkpoint_file = joblib.load(checkpoint_file1)
        except Exception as e:
            raise IOError(f"Failed to load checkpoint file {checkpoint_file1}: {str(e)}")
        
        # Validate checkpoint integrity
        if "pipeline_hash" not in self.checkpoint_file:
            raise ValueError(f"Invalid checkpoint format in {checkpoint_file1}: missing pipeline_hash")
        # print(self.checkpoint_file.keys())
        
        success = self.checkpoint_file.get("success").get("success")
        if not success:
            raise ValueError("This target did not run successfully.")

        # Store all checkpoint components
        self.similarity = self.checkpoint_file.get("results", {}).get("comparison_results")
        self.stats = self.checkpoint_file.get("stats", {})
        #print(self.stats)

    def get_sim_matrix(self,id):
        """returns the similarity matrix with the give id """
        matrix_found = next((d for d in self.similarity if d.get("id") == id), None)
        return matrix_found.get("result",None)

    def get_exp_details(self,cluster_id,clustering_runtime):
        """returns the similarity matrix with the give id """
        cluster_dets = next((d for d in self.options["clustering"] if d.get("id") == cluster_id), None)
        matrix_id =  cluster_dets["matrix_id"]
        matrix_dets = next((d for d in self.options["compare_context"] if d.get("id") == matrix_id), None)
        all_details = {
            "title": self.exp_title, 
            "transform_id": matrix_dets["transformation_id"],
            "matrix_id":matrix_id,
            "clustering_id" : cluster_id,
            "created_on": int( time.time())
        }

        target_stats = self.stats
        time_transform =  next((d["time_s"] for d in target_stats["transform_stats"] if d.get("id") == matrix_dets["transformation_id"]), 0)
        time_compare = next((d["time_s"] for d in target_stats["comparison_stats"] if d.get("id") == matrix_id), 0)
        total = target_stats["timing"]["total"] + time_transform + time_compare + clustering_runtime
        
        config_detail = {
            "transform_id": matrix_dets["transformation_id"],
            "matrix_id":matrix_id,
            "clustering_id" : cluster_id,
            "time_model_input": target_stats["timing"]["model_input"],
            "time_model_train": target_stats["timing"]["model_train"],
            "time_paths_extract": target_stats["timing"]["paths_extract"],
            "time_context_create": target_stats["timing"]["context_create"],
            "time_transform":time_transform,
            "time_compare": time_compare,
            "time_clustering":clustering_runtime,
            "runtime": total
        }
        # print(config_detail)
        return all_details, config_detail
    

    def generate_result_files(self,rerun=False):
        """
        To generate result files for the experiment. A json file is generated for each clustering method specified. 
        """
        for res in self.options["clustering"]:
            print(res)
            file_path = self.checkpoint_dir/ f"{self.target}_res_{res['id']}.json"
            if file_path.exists() and rerun==False:
                print(f"File already exists:{file_path}")
                continue 
            sim_matrix  = self.get_sim_matrix(res["matrix_id"])
            if sim_matrix is None:
                print(f"Sim matrix with id: {res["id"]} not found")
                continue
            #print(sim_matrix)

            ## considering time for just this because storing ans loading pipeline results are just there to make things easier it need not be done necessarily 
            
            start_time = time.perf_counter()    
            all_results = identify_coregulators(
                sim_matrix,
                self.target,
                method= res["method"],
                method_options=res["method_options"]
            )
            end_time = time.perf_counter()
            timing = end_time - start_time

            json_results = to_jsonable(all_results)
            json_results["clustering_runtime"] = timing
            with open(file_path, "w") as f:
                json.dump(json_results, f)
        
    def generate_result(self,rerun=False,include_figures=False,include_stats=True):
        """
        To generate the final result file based on all the clustering among all the targets. 
        """
        result_files  = []
        config_stats = []
        for res in self.options["clustering"]:
            # print(res)
            main_file_path = self.checkpoint_dir/ f"{self.target}_res_{res['id']}.json"
            if main_file_path.exists() is None : 
                # print(f"Main result file not generated :{main_file_path}")
                continue 
            file_path = self.checkpoint_dir/ f"{self.target}_clusters_{res['id']}.csv"
            if file_path.exists() and rerun==False:
                # print(f"File already exists:{file_path}")
                continue 
            with open(main_file_path, "r") as f:
                json_data = json.load(f)
                main_file = from_jsonable(json_data)

            clusters_df = main_file["clusters_df"]
            clusters_df["methodology"] = main_file["methodology"]
            exp_dets,config_dets = self.get_exp_details(res["id"],main_file["clustering_runtime"]) 
            # print(config_dets)
            clusters_df["exp_details"] = format_dict(exp_dets)
            clusters_df.to_csv(file_path,index=False)
            result_files.append(str(file_path))
            config_stats.append(config_dets)
        
        # print(config_stats)
        figure_paths = None
        if include_figures == True:
            figure_paths = self.generate_figures(rerun)
        
        return {"result_paths":result_files,"figure_paths":figure_paths, "config_stats":config_stats }
    
    def get_stats(self):
        """
        Returns stats
        """
        return  self.stats 
        

    def generate_figures(self, rerun: bool = False) -> Dict[str, str]:
        """
        Generate all figures for this target. Creates only missing images.
        """
        import matplotlib.pyplot as plt
        from pathlib import Path
        from coregtor.utils.plot import (
            similarity_matrix_2d_embedding, 
            similarity_matrix_heatmap,
            dendrogram1,
            add_caption
        )
        
        # Create temp figure directory
        figure_dir = self.checkpoint_dir / f"figure_temp"
        figure_dir.mkdir(exist_ok=True)
        
        figures = {}
        
        for sim in self.similarity:
            sim_matrix = sim["result"]  
            sim_name = sim["id"]
            
            # Extract target names
            if hasattr(sim_matrix, 'index') and hasattr(sim_matrix, 'columns'):
                target_names = sim_matrix.index.tolist()
            else:
                n_targets = sim_matrix.shape[0]
                target_names = [f"T{i+1}" for i in range(n_targets)]
            
            figures[sim_name] = {}
            
            # List of figures for this sim matrix
            figure_configs = [
                ("target_similarity_2d", f"{self.target}_{sim_name}_sim_2d.png", 
                lambda: similarity_matrix_2d_embedding(sim_matrix, {"point_size": 250}, target_names)),
                ("target_similarity_heatmap", f"{self.target}_{sim_name}_sim_heatmap.png", 
                lambda: similarity_matrix_heatmap(sim_matrix, {"max_size": 100}, target_names)),
                ("dendrogram", f"{self.target}_{sim_name}_dendrogram.png", 
                lambda: dendrogram1(sim_matrix, {"linkage": "average", "p": 30}, target_names))
            ]
            
            for fig_key, fig_filename, plot_func in figure_configs:
                fig_path = figure_dir / fig_filename
                
                # Check if figure exists and rerun=False
                if not rerun and fig_path.exists():
                    figures[sim_name][fig_key] = str(fig_path)
                    continue
                
                try:
                    # Generate figure
                    # print(f"Generating: {fig_filename}")
                    fig = plot_func()
                    fig = add_caption(fig, f"Similarity matrix {sim_name} for {self.target}")
                    
                    fig.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
                    plt.close(fig)
                    figures[sim_name][fig_key] = str(fig_path)
                    
                except Exception as e:
                    print(f"Failed {fig_filename}: {e}")
                    continue
        
        #print("Done")
        return figures


    # def generate_report():
    #     """
    #     To generate report including stats and figures

    #     figures need to be generated in a temp file and then deleted. this is to save space and the results need to be generated only once        
    #     """
            


def format_dict(d: Dict[str, Any], method: str = "url") -> str:
    """
    Format dictionary in different styles.
    
    Args:
        d: Dictionary to format
        method: "url" (key1=value1&...), "html_ul" (bold keys list), "json", "yaml"
    
    Returns:
        Formatted string
    """
    import urllib.parse
    import json
    import yaml  # Requires PyYAML
    
    def encode_value(v):
        if isinstance(v, (list, tuple)):
            return '&'.join(f"{urllib.parse.quote(str(item))}" for item in v)
        return urllib.parse.quote(str(v))
    
    if method == "url":
        parts = [f"{urllib.parse.quote(str(k))}={encode_value(v)}" for k, v in d.items()]
        return '&'.join(parts)
    
    elif method == "html_ul":
        items = []
        for k, v in d.items():
            if isinstance(v, (list, tuple)):
                v_str = ', '.join(str(item) for item in v)
            else:
                v_str = str(v)
            items.append(f"<li><strong>{k}:</strong> {v_str}</li>")
        return "<ul>" + "".join(items) + "</ul>"
    
    elif method == "json":
        return json.dumps(d, indent=2)
    
    elif method == "yaml":
        return yaml.dump(d, default_flow_style=False, sort_keys=False)
    elif method == "caption":
        # One-line formatted string for plot captions
        items = []
        for k, v in d.items():
            if isinstance(v, (list, tuple)):
                v_str = f"[{', '.join(str(item) for item in v[:3])}{'...' if len(v)>3 else ''}]"
            else:
                v_str = str(v)
            items.append(f"{k}={v_str}")
        return " | ".join(items)
    
    else:
        raise ValueError(f"Unknown method: {method}")

    
        
def _df_to_jsonable(df: pd.DataFrame) -> Any:
  """Convert a DataFrame to a JSON-serializable structure."""
  if df is None:
    return None
  return {
      "columns": list(df.columns),
      "index": df.index.tolist(),
      "data": df.values.tolist(),
  }

def _jsonable_to_df(obj: Any) -> pd.DataFrame:
  """Reconstruct a DataFrame from a JSON-serializable structure."""
  if obj is None:
    return None
  columns = obj.get("columns", [])
  data = obj.get("data", [])
  index = obj.get("index", [])
  return pd.DataFrame(data, columns=columns, index=index)

def to_jsonable(ident_result: Dict[str, Any]) -> Dict[str, Any]:
  """Convert identify_coregulators result to JSON-friendly dict."""
  if ident_result is None:
    return None
  out = {
      "clusters_df": _df_to_jsonable(ident_result.get("clusters_df")),
      "best_cluster": ident_result.get("best_cluster"),
      "methodology": ident_result.get("methodology"),
      "validation_scores": ident_result.get("validation_scores"),
  }
  return out

def from_jsonable(jsonable: Dict[str, Any]) -> Dict[str, Any]:
  """Reconstruct identify_coregulators result from JSON-friendly dict."""
  if jsonable is None:
    return None
  result = {
      "clusters_df": _jsonable_to_df(jsonable.get("clusters_df")),
      "best_cluster": jsonable.get("best_cluster"),
      "methodology": jsonable.get("methodology"),
      "validation_scores": jsonable.get("validation_scores"),
      "clustering_runtime":jsonable.get("clustering_runtime")
  }
  return result
