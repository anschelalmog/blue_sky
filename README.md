Blue Sky: IMU-Based Navigation System

Note: This project is not completed yet.
Status: May 2024
Background

This project explores an IMU-based navigation system for autonomous vehicles using different types of Kalman Filters. The study is conducted at the Faculty of Electrical and Computer Engineering, Technion - Israel Institute of Technology, the Control and Robotics Machine Learning Lab (CRML).
<p align="center">
<img src="assets/trajectory_visualization.png" width="600" height="400" />
</p>
Proposed Model

The system employs Digital Elevation Maps (DEMs) to represent topographical features, which are essential for non-GPS navigation. The navigation system integrates different Kalman Filters to enhance the accuracy of trajectory planning and analysis. Key components include:

    Trajectory Generation: Simulating vehicle motion based on initial conditions and sensor data.
    Pinpoint Finding: Determining the precise location of the vehicle at each point along the generated trajectory.
    Noise Introduction: Applying both bottom-up and top-down approaches to simulate sensor inaccuracies.
    Estimation Algorithms: Utilizing Iterated Extended Kalman Filters (IEKF) and Unscented Kalman Filters (UKF) for state estimation.

<p align="center">
<img src="assets/model_diagram.png" width="600" height="400" />
</p>
Training and Evaluation

The system's performance is evaluated through simulations that include terrain variations and sensor noise. The effectiveness of different Kalman Filters is assessed based on their ability to estimate position, velocity, altitude, and attitude accurately.
<p align="center">
<img src="assets/training_results.png" width="40%" /> <img src="assets/evaluation_results.png" width="40%" />
</p>
Results
Estimation Accuracy
Estimator	Accuracy
IEKF	TBD
UKF	TBD
Files In The Repository
File name	Purpose
src/*.py	Source code for the main algorithms and utility functions
Map/*.py	Map handling and preprocessing scripts
tests/*.py	Unit tests for validating the algorithms
main.py	Main script for running the simulations
run_tests.sh	Bash script to execute all tests
ContUpd_paper.docx	Detailed documentation and report of the project
ContUpd_slides.pptx	Presentation slides summarizing the project
README.md	Project overview and documentation
.gitignore	Specifies files and directories to ignore in version control
Installation

    Clone the repository.
    Install the required packages using the provided environment.yml file by running:

    sh

conda env create -f environment.yml

Alternatively, you can install all the required packages with:

sh

    pip install -r requirements.txt

Prerequisites
Library	Version
Python	3.8+
matplotlib	3.3.4
numpy	1.19.5
scikit_learn	0.24.2
seaborn	0.11.2

Sources & References
Sources

The Digital Elevation Maps (DEMs) were obtained from USGS EarthExplorer.
References

    USGS EarthExplorer
    Bell, B. and Cathey, F. (1993). The iterated Kalman filter update as a Gauss-Newton method. IEEE Transactions on Automatic Control, 38(2), 294-297. doi: 10.1109/9.250476.
    Julier, S., Uhlmann, J., and Durrant-Whyte, H. (1995). A new approach for filtering nonlinear systems. doi: 10.1109/ACC.1995.529783.
    Wan, E. and Van Der Merwe, R. (2000). The unscented Kalman filter for nonlinear estimation. doi: 10.1109/ASSPCC.2000.882463.
