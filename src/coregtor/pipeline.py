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


# Core CoRegTor imports
from coregtor.forest import create_model_input, create_model, tree_paths
from coregtor.context import create_context, transform_context, compare_context
from coregtor.utils.error import CoRegTorError

# Final ID-Driven Schema
SCHEMA = {
    "type": "object",
    "required": [],
    "properties": {
        "target_gene": {
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
                "required": ["id", "method"],
                "additionalProperties": True
            },
            "description": "Context Comparison configs"
        },
        # === PIPELINE CONTROL ===
        "output_dir": {"type": "string", "default": ""},
        "checkpointing": {"type": "boolean", "default": True},
        "force_fresh": {"type": "boolean", "default": False},
        "input":{"type":"object","default":{"expression":"","tflist":""}}
    }
}



class Pipeline:
    def __init__(self,expression_data:pd.DataFrame,tflist:pd.DataFrame,options: Dict[str, Any],exp_title: str = None):
        
        self.expression_data = expression_data
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
            
            self.checkpoint_dir = Path(os.path.expanduser(os.path.expandvars(self.options.get("output_dir")))) / self.title 
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
        
        ## To make the pipeline more efficient, clearing data after success saving of results 
        self.results[target] = None
        self.stats[target] = None
        self.status[target] = None

    
    def _load_checkpoint(self, target: str):
        # to save space, skip loading already finished experiments
        load_old_results = False
        if load_old_results : 
          checkpoint = joblib.load(self._checkpoint_file(target))
          print(f"Loaded checkpoint for {target}")
          self.results[target] = checkpoint["results"]
          self.stats[target] = checkpoint["stats"]

    
    def _auto_discover_targets(self) -> List[str]:
        tfs = set( self.tflist["gene_name"].tolist())
        genes_in_data = set(self.expression_data.columns)
        return sorted(list(tfs & genes_in_data))
    
    def _get_targets(self) -> List[str]:
        targets = self.options.get("target_gene", [])
        if not targets:
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
          print(f"model_input: {timing:.3f}s")
          stats["timing"]["model_input"] = timing
          
          # 3. Model training
          model_config = self.options.get("create_model")
          start_time = time.perf_counter()
          model = create_model(X, Y, **model_config)
          end_time = time.perf_counter()
          timing = end_time - start_time
          print(f"model_train: {timing:.3f}s")
          stats["timing"]["model_train"] = timing
          
          results["model"] = model
          
          # 4. Tree paths
          tree_paths_config = self.options.get("tree_paths")
          start_time = time.perf_counter()
          paths = tree_paths(model, X, Y, **tree_paths_config)
          end_time = time.perf_counter()
          timing = end_time - start_time
          print(f"paths_extract: {timing:.3f}s")
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
          print(f"context_create: {timing:.3f}s")
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
              print(f"transform_{t_config['id']}: {timing:.3f}s ")
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
              print(f"compare_{c_config['id']}: {timing:.3f}s")
              
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
    
    # def run_details(self) -> Dict[str, Any]:
    #     """
    #     Return a JSON-serializable view of the pipeline run.

    #     For each target:
    #     - stats: timing, quality, transform_stats, comparison_stats
    #     - transform_results: [{id, result: 2D list}]
    #     - comparison_results: [{id, result: 2D list}]
    #     """
    #     def _to_serializable_array(obj):
    #         if isinstance(obj, pd.DataFrame):
    #             return {
    #                 "index": obj.index.tolist(),
    #                 "columns": obj.columns.tolist(),
    #                 "values": obj.to_numpy().tolist(),
    #             }
    #         # pandas Series
    #         if isinstance(obj, pd.Series):
    #             return {
    #                 "index": obj.index.tolist(),
    #                 "values": obj.to_numpy().tolist(),
    #             }
    #         # numpy array
    #         if isinstance(obj, np.ndarray):
    #             return obj.tolist()
    #         # fallback: try list() for other array-likes
    #         try:
    #             return list(obj)
    #         except TypeError:
    #             return str(obj)

    #     details: Dict[str, Any] = {"targets": {}}

    #     for target in self.stats.keys():
    #         stats = self.stats[target]
    #         results = self.results[target]

    #         # basic stats (already JSON-friendly with explicit casting) [web:16][web:19]
    #         timing = {k: float(v) for k, v in stats.get("timing", {}).items()}

    #         quality = {}
    #         for k, v in stats.get("quality", {}).items():
    #             if isinstance(v, (int, float, bool, str)):
    #                 quality[k] = v
    #             else:
    #                 try:
    #                     quality[k] = float(v)
    #                 except Exception:
    #                     quality[k] = str(v)

    #         # serialize transform_results
    #         transform_results_serialized: List[Dict[str, Any]] = []
    #         for tr in results.get("transform_results", []):
    #             transform_results_serialized.append(
    #                 {
    #                     "id": tr.get("id"),
    #                     "result": _to_serializable_array(tr.get("result")),
    #                 }
    #             )

    #         # serialize comparison_results
    #         comparison_results_serialized: List[Dict[str, Any]] = []
    #         for cr in results.get("comparison_results", []):
    #             comparison_results_serialized.append(
    #                 {
    #                     "id": cr.get("id"),
    #                     "result": _to_serializable_array(cr.get("result")),
    #                 }
    #             )

    #         details["targets"][target] = {
    #             "timing": timing,
    #             "quality": quality,
    #             "transform_results": transform_results_serialized,
    #             "comparison_results": comparison_results_serialized,
    #         }

    #     return details


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
        default_config["target_gene"] = []
        
        return default_config