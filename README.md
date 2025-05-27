# Few-Shot Taxi-Driver Identification

This project tackles the problem of identifying whether two taxi trajectories belong to the same driver using few-shot learning. Unlike earlier tasks with ample driver data, here we have just 5 days of trajectories from 400 drivers. A Siamese network was used to learn driver-specific embeddings and predict similarity between paired sub-trajectories.

## Contents   
`models/`  
- `siamese_model.pth` - Trained Siamese network weights saved after final epoch  

`python/`   
- `extract_feature.py` - Extracts key trajectory features from raw pickle files
- `generate_paired_traj.py` – Forms labeled (same/different driver) trajectory pairs for training
- `model.py` – Siamese model structure using LSTM-based encoders
- `train.py` – Model training logic
- `test.py` – Testing and inference on validation/test sets
- `main.py` – Main runner script for train / test
- `how-to-run.md` – Quick setup and CLI usage instructions

## Results 

| Metric              | Value                                               |
| ------------------- | --------------------------------------------------- |
| Validation Accuracy | **88.6%**                                           |
| Model Type          | **Siamese LSTM Network**                            |
| Pair Types          | Same-driver: 1, Different-driver: 0                 |
| Features Used       | Longitude, Latitude, Seconds since midnight, Status |
| Best Hyperparams    | LR: 0.001, Batch Size: 64, Epochs: 30               |
