# robocar42

## Installation


Install dependencies:  
pip install --user -r local_req.txt

Run Demo:  
Must be on "robotics" network.

chmod 755 run.sh  
./run.sh

## TODO:
- robocar42:
  * camera.py: OK!
  * car.py: OK!
  * controller.py
    - send_control
    - record_control
  * model.py
    - models
    - generators
    - split
  * preprocess.py
    - check_duplicate
    - augment
    - interpolate
  * util.py
    - view_image
    - occlusion_map
    - configure_log: OK!
    - progress_bar: OK!
- script
  * push_to_cloud.py: OK!
    - takes specified local dataset, interpolate it then push onto gcp bucket
  * staging.py: OK!
    - daemon script. polls staging folder for new files and adds them onto the preprocess queue.
  * preprocessor.py:
    - daemon script. processes items on the preprocess queue then uploads them onto the gcp bucket
  * train.py:
  * drive.py:
- test
  * module test scripts *
