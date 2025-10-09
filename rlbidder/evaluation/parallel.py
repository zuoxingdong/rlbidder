import logging
import multiprocessing as mp
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

import joblib
import polars as pl
from tqdm import tqdm

from rlbidder.constants import DEFAULT_SEED
from rlbidder.evaluation.online import initialize_multi_agents, simulate_multi_agent_campaign

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class ParallelWorker:
    """Worker class that pre-loads and caches data for efficient parallel processing"""
    
    def __init__(
        self,
        data_dir: Path | str,
        period: int,
        seeds: list[int],
        work_dir: Path | str,
        parquet_compression: str,
        history_compress: int,
        cache_dir: Path | str | None,
    ) -> None:
        import os

        import torch

        from rlbidder.evaluation.online import load_delivery_period_auction_data
        
        # Set thread limits to prevent oversubscription
        os.environ.update({
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1", 
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1"
        })
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        
        # Load data once per worker
        self.data = load_delivery_period_auction_data(data_dir, period, verbose=False)
        self.period = period
        # I/O configuration
        self.work_dir = Path(work_dir) / f"period-{period}"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        (self.work_dir / "campaign_reports").mkdir(parents=True, exist_ok=True)
        (self.work_dir / "agent_summaries").mkdir(parents=True, exist_ok=True)
        (self.work_dir / "histories").mkdir(parents=True, exist_ok=True)
        self.parquet_compression = parquet_compression
        self.history_compress = history_compress
        
        # Pre-sample conversions for all seeds
        self.presampled_conversions = {}
        cache_root = Path(cache_dir) if cache_dir is not None else Path(data_dir)
        missing_seeds: list[int] = []
        for seed in seeds:
            cache_path = cache_root / f"presampled_conversions_period-{period}_seed-{seed}.joblib"
            logger.info("Loading presampled conversions from cache: %s", cache_path.name)
            try:
                self.presampled_conversions[seed] = joblib.load(cache_path)
                continue
            except Exception as exc:
                logger.warning(
                    "Failed to load cache %s (seed=%s). Please generate presampled conversions via "
                    "`examples/evaluate_agents.py` or `OnlineCampaignEvaluator` before running the parallel evaluator. "
                    "Error: %s",
                    cache_path.name,
                    seed,
                    exc,
                )
            missing_seeds.append(seed)

        if missing_seeds:
            raise RuntimeError(
                "Missing presampled conversion caches for period "
                f"{period} and seed(s) {missing_seeds}. Run `examples/evaluate_agents.py` (or invoke "
                "`OnlineCampaignEvaluator` once) to materialize caches, then rerun the parallel evaluation."
            )
        
        logger.info("Initialized worker for period %s with %s seeds in %s...", period, len(seeds), self.work_dir)
    
    def run_simulation(self, args: tuple) -> tuple[str, str, str | None]:
        """Execute simulation with pre-loaded data"""
        (
            control_agent_configs,
            all_agent_configs,
            rotate_index,
            seed,
            period,
            budget_ratio,
            cpa_ratio,
            min_remaining_budget,
        ) = args
        
        # Use cached data
        sampled_conversions = self.presampled_conversions[seed]
        
        # Initialize agents
        agents = initialize_multi_agents(
            all_agent_configs=all_agent_configs,
            control_agent_configs=control_agent_configs,
            control_index=rotate_index,
            budget_ratio=budget_ratio,
            cpa_ratio=cpa_ratio,
        )
        
        df_campaign_report, df_agent_summary, auction_history = simulate_multi_agent_campaign(
            agents=agents,
            period=period,
            rotate_index=rotate_index,
            num_advertisers=self.data["num_advertisers"],
            num_timesteps=self.data["num_timesteps"],
            pValues=self.data["pValues"],
            pValueSigmas=self.data["pValueSigmas"],
            sampled_conversions=sampled_conversions,
            budget_ratio=budget_ratio,
            cpa_ratio=cpa_ratio,
            min_remaining_budget=min_remaining_budget,
            verbose=False,
            seed=seed,
        )
        # Persist outputs to disk and return only paths to reduce IPC overhead
        base_name = f"seed-{seed}_rot-{rotate_index}"
        campaign_path = self.work_dir / "campaign_reports" / f"{base_name}.parquet"
        agent_path = self.work_dir / "agent_summaries" / f"{base_name}.parquet"
        df_campaign_report.write_parquet(campaign_path, compression=self.parquet_compression)
        df_agent_summary.write_parquet(agent_path, compression=self.parquet_compression)
        history_path = None
        history_path = self.work_dir / "histories" / f"{base_name}.joblib"
        joblib.dump(auction_history, history_path, compress=self.history_compress)
        # Return small metadata: file paths
        return str(campaign_path), str(agent_path), (str(history_path) if history_path is not None else None)


# Global worker instance (needed for ProcessPoolExecutor)
_worker: ParallelWorker | None = None

def _init_worker_with_class(
    data_dir: Path | str,
    period: int,
    seeds: list[int],
    work_dir: Path | str,
    parquet_compression: str,
    history_compress: int,
    cache_dir: Path | str | None,
) -> None:
    """Initialize global worker instance"""
    global _worker
    _worker = ParallelWorker(data_dir, period, seeds, work_dir, parquet_compression, history_compress, cache_dir)

def _run_simulation_with_worker(args: tuple) -> tuple[str, str, str | None]:
    """Wrapper function for worker simulation"""
    global _worker
    return _worker.run_simulation(args)


class ParallelOnlineCampaignEvaluator:
    def __init__(self, data_dir: Path, min_remaining_budget: float = 0.1, verbose: bool = True) -> None:
        self.data_dir = data_dir
        self.min_remaining_budget = min_remaining_budget
        self.verbose = verbose

    def evaluate(
        self,
        control_agent_configs: tuple,
        all_agent_configs: list[tuple],
        delivery_period_indices: list[int] | None = None,
        budget_ratio: float | None = None,
        cpa_ratio: float | None = None,
        seeds: list[int] = [DEFAULT_SEED],
        num_workers: int | None = None,
        chunksize: int | None = None,
        work_dir: Path | None = None,
        keep_artifacts: bool | None = None,
        parquet_compression: str = "zstd",
        history_compress: int = 3,
        streaming: bool = False,
        cache_dir: Path | None = None,
    ) -> tuple[pl.DataFrame, pl.DataFrame, list, None]:
        if num_workers is None:
            num_workers = min(mp.cpu_count() - 1, len(seeds) * (len(all_agent_configs) + 1))
            num_workers = max(1, num_workers)
        
        if chunksize is None:
            total_jobs_per_period = len(seeds) * (len(all_agent_configs))
            chunksize = max(1, total_jobs_per_period // (num_workers * 4))
        
        if delivery_period_indices is None:
            delivery_period_indices = (
                pl.scan_parquet(self.data_dir / "eval-period-*.parquet")
                .select("deliveryPeriodIndex")
                .unique()
                .sort("deliveryPeriodIndex")
                .collect()
                .get_column("deliveryPeriodIndex")
                .to_list()
            )
        # Prepare working directory
        created_tmp = False
        if work_dir is None:
            tmp_dir = Path(".tmp/")
            tmp_dir.mkdir(parents=True, exist_ok=True)
            root_work_dir = Path(tempfile.mkdtemp(prefix="rlbidder_parallel_", dir=tmp_dir))
            logger.info("Created temporary work dir: %s", root_work_dir)
            created_tmp = True
        else:
            root_work_dir = Path(work_dir)
            root_work_dir.mkdir(parents=True, exist_ok=True)
        if keep_artifacts is None:
            keep_artifacts = not created_tmp

        campaign_report_paths = []
        agent_summary_paths = []
        history_paths = []

        for period in delivery_period_indices:
            rotate_indices = range(len(all_agent_configs))
            
            jobs = [
                (
                    control_agent_configs,
                    all_agent_configs,
                    rotate_index,
                    seed,
                    period,
                    budget_ratio,
                    cpa_ratio,
                    self.min_remaining_budget,
                )
                for seed, rotate_index in product(seeds, rotate_indices)
            ]
            logger.info("Running %s jobs for period %s in %s...", len(jobs), period, root_work_dir)
            
            with ProcessPoolExecutor(
                max_workers=num_workers,
                mp_context=mp.get_context("spawn"),  # safer with SciPy/BLAS on Linux
                initializer=_init_worker_with_class,
                initargs=(
                    self.data_dir, 
                    period, 
                    seeds, 
                    str(root_work_dir), 
                    parquet_compression, 
                    history_compress, 
                    cache_dir,
                )
            ) as executor:
                if len(jobs) > 100:
                    future_to_job = {executor.submit(_run_simulation_with_worker, job): job for job in jobs}
                    
                    results = []
                    for future in tqdm(
                        as_completed(future_to_job), 
                        total=len(jobs),
                        desc=f"Period {period}",
                        disable=not self.verbose
                    ):
                        try:
                            result = future.result()
                            results.append(result)
                        except Exception:
                            job = future_to_job[future]
                            logger.exception("Job %s generated an exception", job)
                else:
                    results = list(tqdm(
                        executor.map(_run_simulation_with_worker, jobs, chunksize=chunksize),
                        total=len(jobs),
                        desc=f"Period {period}",
                        disable=not self.verbose,
                    ))

            for campaign_path, agent_path, history_path in results:
                campaign_report_paths.append(Path(campaign_path))
                agent_summary_paths.append(Path(agent_path))
                if history_path is not None:
                    history_paths.append(Path(history_path))
            
            del results

        # Merge results efficiently in parent process
        if streaming:
            lf_campaigns = (
                pl.concat([pl.scan_parquet(str(p)) for p in campaign_report_paths])
                if campaign_report_paths
                else None
            )
            lf_agents = (
                pl.concat([pl.scan_parquet(str(p)) for p in agent_summary_paths])
                if agent_summary_paths
                else None
            )
            df_campaign_reports = (
                lf_campaigns.collect(streaming=True)
                if lf_campaigns is not None
                else pl.DataFrame([])
            )
            df_agent_summaries = (
                lf_agents.collect(streaming=True)
                if lf_agents is not None
                else pl.DataFrame([])
            )
        else:
            df_campaign_reports = (
                pl.concat([pl.read_parquet(str(p)) for p in campaign_report_paths], rechunk=True)
                if campaign_report_paths
                else pl.DataFrame([])
            )
            df_agent_summaries = (
                pl.concat([pl.read_parquet(str(p)) for p in agent_summary_paths], rechunk=True)
                if agent_summary_paths
                else pl.DataFrame([])
            )

        # Handle histories per user preference
        auction_histories: list = []
        for hp in history_paths:
            auction_histories.append(joblib.load(hp))

        # Cleanup temporary artifacts if created here and not requested to keep
        if created_tmp and not keep_artifacts:
            shutil.rmtree(root_work_dir, ignore_errors=True)

        return df_campaign_reports, df_agent_summaries, auction_histories, None
