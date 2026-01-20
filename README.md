Will update the README after i finish implementing everything 

Short Description: 
Task Scheduling Using a trained Actor-Critic Model and a GAT based understanding and importance measuring of different task states that needs to be scheduled.


TODO: 
- Need to implement a communication interface (Like All to All). Right now only on one cluster of machines.  

- Need to implement a MCTS or "plan ahead" algorithms. Very important -- Done with the MCTS algorithm code , need to test it and fix obvious errors. 


- Change from LSTM pointer networks to transformers 

- Add prometheus

- Add database framework. Architecture is tier based with a low tier for an long record of jobs and job related data, Postgres for this, and a high tier storage like fred.rs to be used for MCTS data storage since that will ensure low latency.

- Consistent env framework . Need to map the structs properly from actor output to the env.

- A few good AI suggestions: 
    - No error recovery in environment
    - No rollback mechanism for failed scheduling
    - No resource deadlock detection
    - No graceful degradation when clusters fail
    - No checkpointing for training state
    - No gradient clipping or normalization
    - No experience replay buffer for A2C 
- will work on all of this properly
