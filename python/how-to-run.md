## How to Run

extract features from the pickle file and generate the sub-trajectories of the 400 drivers:   
- $ python extract_feature.py  

generate pairs of sub-trajectories where pairs originating from the same driver are labeled with a 1 and pairs from different drivers are labeled with a 0. Upon executing this script, the expected shape of the training dataset will be (number of trajectory pairs, 2, 100, feature size).
- $ generate_paired_traj.py
  
training model:
- $ python main.py train
  
testing model:
- $ python main.py test  
