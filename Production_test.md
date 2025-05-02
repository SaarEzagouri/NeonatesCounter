## Production test

#### 5.2.2025
**Model *conf* increased from 0.045 --> 0.65**. The difference between the output of the MATLAB and Python algorithms was ~20% therefore the conf was increased. Revise the conf after the next validation round comparing the 2 algorithms and Samuel's annotation. <br>
**Windows scheduler bug fix:** the reason for the bug is path of the .py file was in dropbox. I switched to working locally, now the path to the image dir is hard coded and not deduced. The scheduler is set for daily run at 23:00.
