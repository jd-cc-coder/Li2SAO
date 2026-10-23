# Contributing to slime

Thank you for your interest in contributing to slime! We deeply appreciate every contribution from the community. To keep the project healthy and sustainable, please read this document carefully before submitting issues or pull requests.

## Collaboration Scope

slime is the RL training infrastructure behind [GLM-4.5 through GLM-5.2](https://z.ai) and a large number of internal experiments at Z.ai. We open-sourced slime because we believe the training scenarios used internally cover the majority of cutting-edge RL algorithm requirements, and we hope to provide the community with a correct and efficient large-scale RL training infrastructure.

Our goal for open-source collaboration is focused on **bug fixes** and **general-purpose large-scale RL optimizations**. We have had several successful collaborations with the community in this area, including:

- Speculative decoding in RL ([docs](https://thudm.github.io/slime/en/advanced/speculative_decoding.html))
- Low-precision training: fp8 rollout + bf16/fp8 training, int4 rollout + int4 QAT training ([docs](https://thudm.github.io/slime/en/advanced/low_precision_training.html))
- Deterministic training ([docs](https://thudm.github.io/slime/en/advanced/reproducibility.html))

### What We Welcome

| Category | Examples |
|----------|----------|
| **Bug reports** | Crashes, incorrect results, documentation errors |
| **Bug fixes** | PRs that fix existing issues with tests or clear reproduction |
| **General RL optimizations** | Performance improvements with clear benchmarks that can be verified through CI or standard training runs |

### What's Currently Outside Our Scope

| Category | Reason |
|----------|--------|
| **Large-scale code refactoring** | This would add considerable overhead to syncing between the internal and open-source versions, particularly in coordinating with internal algorithm teams. |
| **Design / abstraction proposals** | e.g., universal data standards, eval standards, tool base classes. Standard-setting involves non-technical factors; slime intentionally avoids such content to keep things flexible for both the community and internal teams. |
| **Features that cannot be clearly verified** | Correctness is critically important for a training framework. If a feature cannot be verified through CI or routine internal training, it becomes difficult for us to ensure timely fixes, which could affect the project's long-term reliability. |
| **Features independent of the RL framework** | e.g., full algorithm reproduction pipelines. While these lower the barrier to entry, they are difficult to include in routine verification. slime aims to be lightweight — more like Flask than Django. We recommend building such pipelines in separate repositories; we are happy to reference them in the README. |
| **Major modifications to Megatron** | We do not plan to maintain a Megatron fork through slime. The goal is to switch Megatron versions relatively painlessly; Megatron performance optimization and feature completion are not primary objectives. |

### Why This Policy?

slime's design and development roadmap must first align with Z.ai's internal requirements — the RL infrastructure design is tightly coupled with post-training plans, and publishing the full post-training R&D roadmap is not within the open-source scope of slime. This focused scope is what we believe is necessary to maintain slime as a long-term, trustworthy project.

We understand this may slow down slime's pace of feature development. We are actively expanding the slime team through hiring to help with this — if you're interested, feel free to reach out directly.

Thank you for your understanding and patience. We truly appreciate the effort community contributors put in, and we're sorry if this policy causes any inconvenience.
