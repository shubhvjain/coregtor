import json
import gzip
import time
import joblib
import jsonschema
import pandas as pd
import traceback
from pathlib import Path
from typing import Dict, Any, List
import hashlib
import os

from coregtor.forest import create_model, tree_paths
from coregtor.context import create_context, transform_context, compare_context, get_distance_measures_list
from coregtor.utils.error import CoRegTorError

from coregtor.clusters import identify_coregulators, get_cluster_method_list

from joblib import Parallel, delayed, cpu_count

# keys that affect computation — used for checkpoint hash
COMPUTE_KEYS = ["create_model", "tree_paths",
                "create_context", "transform_context", "compare_context"]

SCHEMA = {
    "type": "object",
    "properties": {
        "target_genes": {
            "type": "array",
            "items": {"type": "string"}
        },
        "ensemble_model": {
            "type": "object",
            "properties": {
                "method": {"type": "string", "enum": ["rf", "et"], "default": "rf"},
                "options": {"type": "object", "default": {"max_depth": 5, "n_estimators": 1000}}
            }
        },
        "tree_paths": {"type": "object", "default": {}, "description": "Options related to generation of root to leaf tree paths. No options required at the moment"},
        "context_generation": {
            "description": "To describe the method to generate the context for unique root nodes",
            "type": "object",
            "properties": {
                "method": {
                    "type": "string", "default": "tree_paths"
                },
                "output": {
                    "type": "string", "default": "gene_frequency"
                }
            },
            "default": {"method": "tree_paths", "output": "gene_frequency"}
        },
        "context_comparison": {
            "description": "Distance measures to use. ",
            "type": "object",
            "properties": {
                "methods": {
                    "type": "array",
                    "uniqueItems": True,
                    "minItems": 1,
                    "items": {
                        "type": "string",
                        "enum": get_distance_measures_list()
                    }
                },
            },
            "default": ["cosine_distance"],
            "additionalProperties": True
        },
        "clustering": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "distance", "method"],
                "properties": {
                    "id": {"type": "string", "default": "default"},
                    "distance": {"type": "string", "default": "cosine_distance"},
                    "method": {
                        "type": "string",
                        "enum": get_cluster_method_list(),
                        "default": "hierarchical"
                    },
                    "options": {"type": "object", "default": {"auto_threshold": "inconsistency"}}
                },
                "additionalProperties": True
            }
        },
        "run": {
            "description": "These options handle the running of pipeline",
            "type": "object",
            "properties": {
                "checkpointing": {"type": "boolean", "default": True},
                "save_model": {"type": "boolean", "default": False, "description": ""},
                "save_tree_paths": {"type": "boolean", "default": False},
                "rerun": {"type": "boolean", "default": False},
                "temp_path": {"type": "string", "default": ""},
                "output_path": {"type": "string", "default": ""},
                "create_folder": {"type": "boolean", "default": False, "description": "If true, a new folder will be created inside the temp and output path with the provided exp_title "}
            },
            "additionalProperties": True
        }
    }
}


def apply_defaults(data, schema):
    """Recursively apply default values from schema to data where keys are missing."""
    if not isinstance(schema, dict):
        return data

    schema_type = schema.get("type")

    if schema_type == "object":
        if not isinstance(data, dict):
            return data
        properties = schema.get("properties", {})
        for key, prop_schema in properties.items():
            if key not in data:
                if "default" in prop_schema:
                    data[key] = prop_schema["default"]
            else:
                data[key] = apply_defaults(data[key], prop_schema)

    elif schema_type == "array":
        if not isinstance(data, list):
            return data
        item_schema = schema.get("items", {})
        if not isinstance(data, list):
            return data
        # apply defaults to each existing item, but do NOT add default items to the array
        data = [apply_defaults(item, item_schema) for item in data]
        # if the array itself has a top-level default and is empty, leave it — caller decides

    return data


def get_validated_options(options):
    # 1. validate structure first
    jsonschema.validate(instance=options, schema=SCHEMA)

    # 2. check unique cluster ids
    cluster_ids = [c["id"] for c in options.get("clustering", []) if "id" in c]
    if len(cluster_ids) != len(set(cluster_ids)):
        duplicates = {id for id in cluster_ids if cluster_ids.count(id) > 1}
        raise jsonschema.ValidationError(
            f"Duplicate cluster ids found: {duplicates}")

    # 3. fill in missing defaults
    import copy
    result = copy.deepcopy(options)
    apply_defaults(result, SCHEMA)

    return result


def get_default_options():
    result = {}
    apply_defaults(result, SCHEMA)
    return result


class CoRegTorPipeline:
    def __init__(self, expression_data, source_genes, options, exp_title=None):

        if expression_data is None:
            raise ValueError("no expression data provided")
        if not source_genes:
            raise ValueError("provide a list of source genes to use a input")

        self.expression_data = expression_data
        self.source_genes = source_genes

        # defaults = CoRegTorPipeline._generate_default_config_dict()
        # options = {**defaults, **(options or {})}
        # jsonschema.validate(instance=options, schema=SCHEMA)
        self.options = get_validated_options(options)
        # print(self.options)
        self.results = {}
        self.stats = {}
        self.status = {}

        self.title = exp_title.replace(
            " ", "_") if exp_title else f"exp_{int(time.time())}"

        # create a folder in  temp locations if required
        temp_path = self.options["run"]["temp_path"]
        if self.options["run"]["create_folder"]:
            temp_path = f"{temp_path}/{self.title}"

        self.checkpoint_dir = Path(temp_path)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        out_path = self.options["run"]["output_path"]
        if self.options["run"]["create_folder"]:
            out_path = f"{temp_path}/{self.title}"

        self.output_dir = Path(out_path)

        # self.targets = targets  # List of targets to process

        self.X_shared = None
        self.feature_columns = None
        self._prepare_shared_X()

    def _prepare_shared_X(self):
        self.feature_columns = [
            g for g in self.source_genes if g in self.expression_data.columns]
        if not self.feature_columns:
            raise ValueError("no valid feature columns found")
        self.X_shared = self.expression_data[self.feature_columns]

    def _get_model_input(self, target):
        if target not in self.expression_data.columns:
            raise ValueError(f"target '{target}' not in expression data")
        Y = self.expression_data[[target]]
        if target in self.feature_columns:
            X_cols = [col for col in self.feature_columns if col != target]
            X = self.X_shared[X_cols]
        else:
            X = self.X_shared
        return X, Y

    def _checkpoint_file(self, target: str) -> Path:
        return self.checkpoint_dir / f"{target}.pkl"

    def _checkpoint_exists(self, target: str) -> bool:
        if not self.options.get("checkpointing", True) or self.options.get("force_fresh", False):
            return False
        f = self._checkpoint_file(target)
        if not f.exists():
            return False
        else:
            return True

    def _save_checkpoint(self, target: str):
        checkpoint = {
            "timestamp": time.time(),
            "results": self.results[target],
            "stats": self.stats[target],
            "success": self.status[target]
        }
        joblib.dump(checkpoint, self._checkpoint_file(target), compress=3)
        self.results[target] = None
        self.stats[target] = None
        self.status[target] = None

    def run_single_target(self, target, rerun=False):
        """Run pipeline for one gene. Raises on failure."""
        if self._checkpoint_exists(target) and not rerun:
            return

        # Total available CPUs (including logical cores)
        #print(cpu_count()) 
        # Total physical cores
        #print(cpu_count(only_physical_cores=True)) 

        stats = {"timing": {}, "quality": {}, "n_cores":cpu_count()}
        results = {}
        status = {"success": False, "error": ""}

        save_model = self.options["run"].get("save_model", False)
        save_paths = self.options["run"].get("save_tree_paths", False)

        try:
            if target not in self.expression_data.columns:
                raise ValueError(f"target '{target}' not in expression data")

            t = time.perf_counter()
            X, Y = self._get_model_input(target)
            stats["timing"]["model_input"] = time.perf_counter() - t

            t = time.perf_counter()
            model = create_model(X, Y, **self.options.get("ensemble_model"))
            stats["timing"]["model_train"] = time.perf_counter() - t
            if save_model:
                results["model"] = model

            t = time.perf_counter()
            paths = tree_paths(model, X, Y, **self.options.get("tree_paths"))
            stats["timing"]["paths_extract"] = time.perf_counter() - t
            stats["quality"]["n_paths"] = len(paths)
            stats["quality"]["n_unique_roots"] = paths["source"].nunique()
            if save_paths:
                results["paths"] = paths

            context_options = self.options.get("context_generation")

            t = time.perf_counter()
            contexts = create_context(
                paths, method=context_options.get("method"))

            transformed = transform_context(
                contexts, method=context_options.get("output"))

            stats["timing"]["context_create"] = time.perf_counter() - t
            stats["quality"]["n_contexts"] = len(contexts)

            results["context"] = contexts
            results["gene_frequency"] = transformed

            comparison_results = {}

            comparison_methods = self.options.get(
                "context_comparison").get("methods")

            for m in comparison_methods:
                matrix = compare_context(
                    transformed, method=m, transformation_type="gene_frequency")
                comparison_results[m] = matrix

            results["distance"] = comparison_results

            # clustering results
            cluster_results = {}
            cluster_options = self.options.get("clustering")
            for c in cluster_options:
                id = c["id"]
                dist_matrix = comparison_results[c["distance"]]
                c_method = c["method"]
                c_options = c["options"]
                c_notes = c["note"]
                res = identify_coregulators(
                    dist_matrix, target, c_method, c_options, c_notes)
                cluster_results[id] = res

            results["clusters"] = cluster_results

            stats["timing"]["total"] = sum(stats["timing"].values())
            status["success"] = True
            self.results[target] = results
            self.stats[target] = stats
            self.status[target] = status

            if self.options.get("checkpointing", True):
                self._save_checkpoint(target)

        except CoRegTorError as e:
            status["success"] = False
            status["error"] = str(e)
            self.results[target] = None
            self.stats[target] = None
            self.status[target] = status
            if self.options.get("checkpointing", True):
                self._save_checkpoint(target)
            raise

    def generate_all_clusters(self, targets, cluster_id="default"):
        """
        """
        


class PipelineResults:
    def __init__(self, options: Dict[str, Any], tflist: list, exp_title: str = None, targets: List[str] = None):
        self.tflist = tflist
        jsonschema.validate(instance=options, schema=SCHEMA)
        self.options = options
        self.title = exp_title
        self.checkpoint_dir = Path(
            os.path.expandvars(self.options["paths"]["temp"]))
        self.output_dir = Path(os.path.expandvars(
            self.options["paths"]["output"]))
        self.targets = targets  # List of targets to process

    def get_successful_targets(self) -> List[str]:
        """Get list of targets to process.

        If targets were explicitly provided, use those.
        Otherwise, scan checkpoint directory for pkl files.
        """
        if self.targets is not None:
            # Filter to only include targets that have pkl files
            available = {f.stem for f in self.checkpoint_dir.glob("*.pkl")}
            filtered = [t for t in self.targets if t in available]
            if len(filtered) != len(self.targets):
                missing = set(self.targets) - available
                print(
                    f"Warning: {len(missing)} targets missing pkl files: {list(missing)[:5]}...")
            return filtered
        else:
            # Fallback: scan directory for pkl files
            return [f.stem for f in self.checkpoint_dir.glob("*.pkl")]

    def generate_clusters_file(self):
        targets = self.get_successful_targets()
        n_jobs = self.options.get("result_generation", {}).get("n_jobs", 4)
        rerun_files = self.options.get(
            "result_generation", {}).get("rerun", False)
        print(f"processing {len(targets)} targets with {n_jobs} workers")

        methods_set = self.options.get("clustering", [])
        if len(methods_set) == 0:
            raise ValueError("No clustering method specified")

        for m in methods_set:
            # check if already created

            method_id = m["id"]
            print(f"Processing {method_id}")
            out = self.output_dir / f"result_{method_id}.csv"
            if out.exists() and not rerun_files:
                print("Result file already exists")
                continue
            all_results = Parallel(n_jobs=n_jobs, verbose=0)(
                delayed(self._process_single_target)(target, m)
                for target in targets
            )

            valid_results = [r for r in all_results if r is not None]
            print(f"Valid results: {len(valid_results)} / {len(all_results)}")

            # Extract best clusters
            all_best_results = [
                r["best_cluster"]
                for r in valid_results
                if r["best_cluster"] is not None
            ]

            # Save best clusters
            combined = pd.DataFrame(all_best_results)
            combined.to_csv(out, index=False)
            print(f"Saved {len(combined)} rows to {out}")

            # Save similarity matrices
            save_sim = self.options.get("save_similarity_matrix", False)
            if save_sim:
                sim_out = self.output_dir / f"sim_{method_id}.json.gz"
                if not sim_out.exists():
                    sims = {
                        item["target"]: item["sim_matrix"].to_dict(
                            orient='index')
                        for item in valid_results
                        if item["sim_matrix"] is not None
                    }
                    with gzip.open(sim_out, 'wt') as f:
                        json.dump(sims, f, indent=0)

            # Save all clusters
            save_all_clusters = self.options.get("save_all_clusters", False)
            if save_all_clusters:
                all_clusters_list = [
                    item["all_clusters"]
                    for item in valid_results
                    if item["all_clusters"] is not None and not item["all_clusters"].empty
                ]
                if all_clusters_list:
                    combined_df = pd.concat(
                        all_clusters_list, ignore_index=True)
                    out_all_c = self.output_dir / \
                        f"clusters_{method_id}.parquet"
                    combined_df.to_parquet(
                        out_all_c, compression='gzip', index=False)

    def _process_single_target(self, target: str, method_config: str) -> Dict[str, pd.DataFrame]:
        try:
            checkpoint_file = self.checkpoint_dir / f"{target}.pkl"
            if not checkpoint_file.exists():
                return None

            checkpoint_data = joblib.load(checkpoint_file)

            results = {
                "all_clusters": None,
                "best_cluster": None,
                "sim_matrix": None,
                "sim_matrix_name": method_config["matrix_id"],
                "target": target
            }

            method_id = method_config["id"]
            sim_matrix = self._get_sim_matrix(
                checkpoint_data, method_config["matrix_id"])
            results["sim_matrix"] = sim_matrix
            if sim_matrix is None:
                return None

            _, clusters_df, best_cluster = identify_coregulators(
                sim_matrix,
                target,
                method=method_config["method"],
                method_options=method_config["method_options"]
            )
            if clusters_df is None or clusters_df.empty:
                print(f"No clusters generated for {target}")
                return None

            clusters_df["note"] = f"{self.title}-coregtor-{method_id}"
            if best_cluster is not None:
                best_cluster["note"] = f"{self.title}-coregtor-{method_id}"
            best_cluster["note"] = f"{self.title}-coregtor-{method_id}"
            results["all_clusters"] = clusters_df
            results["best_cluster"] = best_cluster
            return results
        except Exception as e:
            traceback.print_exc()
            print(f"error processing {target}: {e}")
            return None

    def _get_sim_matrix(self, checkpoint_data: Dict, matrix_id: str):
        results = checkpoint_data.get(
            "results", {}).get("comparison_results", [])
        found = next((d for d in results if d.get("id") == matrix_id), None)
        return found.get("result") if found else None
