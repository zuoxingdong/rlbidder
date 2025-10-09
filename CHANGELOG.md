# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-10-09

### Added
- Initial release of rlbidder
- Rust-powered data pipeline with Polars Lazy API for processing massive auction logs
- SOTA offline RL algorithms: IQL, CQL, Decision Transformer, GAVE, GAS
- Classic control baselines: BudgetPacer, PID, Fixed CPA, Stochastic CPA
- Modern transformer stack with FlashAttention (SDPA), RoPE, QK-Norm, SwiGLU
- HL-Gauss distributional RL implementation
- Efficient ensemble critics using torch.vmap (faster than loop-based)
- Numerically stable stochastic policies (SigmoidRangeStd, BiasedSoftplus)
- Parallel online evaluation with ProcessPool and round-robin agent rotation
- Interactive Plotly dashboards for campaign monitoring and market analytics
- RLiable metrics for statistically robust algorithm comparison
- PyTorch Lightning training infrastructure with multi-GPU support
- Draccus type-safe dataclass-to-CLI configuration management
- Local experiment tracking with Aim
- Decision Transformer inference buffer for online deployment
- Comprehensive data preprocessing pipeline with scikit-learn-style transformers
- Production-ready auction simulator with multi-agent support

### Features
- **Data Processing**: Symlog, Winsorizer, ReturnScaledReward transformers
- **Evaluation**: Multi-seed, multi-period evaluation with parallel workers
- **Visualization**: Campaign health, budget pacing, market structure, scatter analysis
- **Scalability**: Stream processing, mixed precision training, gradient accumulation

[0.1.0]: https://github.com/zuoxingdong/rlbidder/releases/tag/v0.1.0