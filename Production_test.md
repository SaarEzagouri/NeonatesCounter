## Production test

Root folder: "C:\Saar\Projects\NeonatesCounter" <br>
NeonatesCounter Directory Structure: <br>

```plaintext
NeonatesCounter/
├── Dev/
│   ├── model.pt
│   ├── file1.py
│   └── file2.py
└── Production/
    ├── model.pt
    ├── file1.py
    └── file2.py
```

#### 5.2.2025
**Model *conf* increased from 0.045 --> 0.65**. The difference between the output of the MATLAB and Python algorithms was ~20% therefore the conf was increased. Revise the conf after the next validation round comparing the 2 algorithms and Samuel's annotation. <br>
**Windows scheduler bug fix.** the reason for the bug is path of the .py file was in dropbox. I switched to working locally, now the path to the image dir is hard coded and not deduced. The scheduler is set for daily run at 23:00. <br>

**Tests:** <br>
:white_check_mark: The code is analyzing every image in all dirs created, but not modified, on the day of the running. <br>
:white_check_mark: All analyzed images are saved. skips Overlay.jpg. updating the excel. <br>
:white_check_mark: Scheduler is running with forced run. <br>
✅ Error Handling: No excel file and/or images found, display error message and continue. <br>
✅ Scheduler should work at 23:00 at both locations, even if windows is logged out. <br>

**To Improve:** <br>
Error handling: structure function in try - except.

