# FACC

[![DOI](https://zenodo.org/badge/598025346.svg)](https://zenodo.org/badge/latestdoi/598025346)

- We propose a GD allocation model with the objectives of maximizing guaranteed delivery and user interest and minimizing the penalty of extra traffic cost simultaneously. Our model is robust for guaranteed delivery because we impose soft rather than hard constraints on the traffic cost of each GD contract.

- We also propose an online bidding strategy based on the allocation model to generate the optimal online allocation solution, which is simple to calculate and more adaptive to dynamic unstable traffic environments during the online serving stage. The strategy dynamically adjusts the weights for each GD contract between impressions’ quality and traffic cost based on real-time performance, which brings fairness-aware allocation results.

## FACC Model
- See details in the paper “Fairness-aware Guaranteed Display Advertising Allocation under Traffic Cost Constraint”.

## Dataset
- **Note:** In paper, all algorithms are trained based on a distributed Parameter-Server architecture because they are estimated on a large-scale dataset with 36 million real-world requests. Here to demostrate the algo clearly, a synthetically generated dataset is used since there is no public GD dataset.

## Run Environment
- This software suite has been tested under:
  - Apple macOS Mojave; python3.8

## Citation

- If you find this code useful for your research, please consider citing:

  - `Fairness-aware Guaranteed Display Advertising Allocation under Traffic Cost Constraint`

## License
Apache-2.0
