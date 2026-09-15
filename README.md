# NeuroAI Task Similarity Project

This project studies how task similarity affects learned neural representations in a simple synthetic multi-task setting.

## Current setup

Input data:
- 1000 samples
- 4 features
- Features sampled from a standard normal distribution

Tasks:
- Task A: based on x1 and x2
- Task B high similarity
- Task B medium similarity
- Task B low similarity

Observed label agreement with Task A:
- B high: 0.906
- B medium: 0.663
- B low: 0.508

## Files

- `generate_data.py`: generates input data and task labels
- `check_task_similarity.py`: checks label agreement between tasks
- `split_data.py`: creates train, validation, and test splits

## Next step

Build and train the first neural network model.