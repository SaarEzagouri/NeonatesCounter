# NeonatesCounter
Counting neonates of black soldier flies in high resolution images.

### Summary:
NeonatesCounter_v1.0.py processes an input image by dividing it into smaller patches, detecting neonates in each patch, and then stitching the patches back together to reconstruct the full image, adding padding if necessary. <br>
The script outputs images with corresponding bounding boxes each analyzed image (Figure 1). This script returns a dictionary with the structure: `{"image_name": number_of_detected_objects}`, which can be used by subsequent code when running the `NeonatesCounter_v1.0.py` file from within MATLAB.  <br><br>

![](schematic_pipeline.png)
**Figure 1.** A schematic representation of the algorithm. <br><br><br><br>

<p align="center">
  <img src="FreezeMLogo.png" alt="FreezeM Logo">
</p>
<p align="center">
  <strong>Version 1.0, 3/2025, All rights reserved to <a href="https://www.freezem.com/">FreezeM</a></strong>
</p>
<p align="center">
  Development and maintenance by Saar Ezagouri
</p>
