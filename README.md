# About the paper

In low-altitude urban environments, unmanned  aerial vehicles (UAVs) demand reliable localization and motion tracking to sustain stable communication links. Conventional base-station–centric sensing, however, is susceptible to blockage and environmental clutter, leading to degraded localization ac curacy and reduced link reliability. To overcome these limita tions, we introduce a distributed-camera, multimodal integrated sensing and communication (ISAC) framework that enables omnidirectional UAV perception. Leveraging time-synchronized camera arrays, the proposed architecture provides full-viewpoint UAV localization and effectively eliminates sensing blind zones. The camera-driven perception module further enhances UAV motion prediction, thereby facilitating proactive beamforming and improving end-to-end communication performance. A multimodal state estimation and prediction model is developed to ensure robust UAV tracking under complex urban conditions. To validate the framework, we construct a realistic simulation environment on the Genesis robotics platform, generating diverse, temporally aligned multimodal datasets for training and per formance assessment. Experimental results demonstrate that the proposed approach substantially improves sensing robustness and achievable communication rate relative to conventional single viewpoint sensing schemes.



1. Simulation Platform
Simulation platform: https://github.com/Genesis-Embodied-AI/Genesis
Download the Genesis simulation platform.
Place data_gen.py in the root directory and execute it.
The script generates:
Image data
UAV trajectory data
All generated data are stored in the Data2 folder.
2. Data Processing
Use read_mp4.py in the Data2 folder to perform fast object detection on video data.
UAV trajectory data are imported into MATLAB for:
Echo signal simulation
Echo-based algorithm processing
All processed data are stored in the cache2 folder.
   
3.plot

plot.py and plot_cpf.py


<img width="999" height="419" alt="image" src="https://github.com/user-attachments/assets/1d716f46-9074-4061-9de3-70f919cbceed" />

<img width="767" height="1148" alt="image" src="https://github.com/user-attachments/assets/7f73d3b7-c055-478c-9e26-efeeba26444f" />



