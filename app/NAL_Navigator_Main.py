####################################################################################################
##################################### NAL Navigator ################################################
####################################################################################################
# Authors = Luke Jorgensen and Joshua Jorgensen
# Version = 0.0.1

# Build Command: C:\Users\tajor>auto-py-to-exe -bdo "C:\Users\tajor\Python_Build_Dir\Build_Folder"
## Loading Needed Libareies
import subprocess
import time

print(" Running the Streamlit App")
#time.sleep(5)
## Running the Streamlit NAL Navigator
subprocess.run(["streamlit", "run", "NAL_Navigator.py"])
#time.sleep(5)
print(" Exiting Subprocess")
#time.sleep(5)
