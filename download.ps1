$url = 'https://github.com/robaita/introduction_to_machine_learning/raw/main/dataset.zip'  
Invoke-WebRequest -Uri $url -OutFile 'dataset.zip'  
Expand-Archive -Force -Path 'dataset.zip' -DestinationPath '.'  
