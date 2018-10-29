# cities-dl
This is a repo containing an image clasification dash app that is able to recognize what city is visible on the image.
Behind the app there is a CNN network (VGG16) fine tuned to recognize cities.

1. Create virtual env and activate it.
```
conda create -n myenv python=3.5
```
2. Install all required packages.
```
conda config --append channels conda-forge
conda config --append channels menpo
conda install --yes --file conda_requirements.txt
```
3. cd into the directory of the repo. Run the app.
```
python app.py
```

![]("app_image.PNG")
