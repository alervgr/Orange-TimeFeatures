from setuptools import setup

setup(name="TimeFeatures",
      packages=["timefeatures.widgets"],
      package_data={"timefeatures.widgets": ["icons/*.svg", "icons/*.png"]},
      entry_points={"orange.widgets": "Time-Features = timefeatures.widgets"},
      version="1.0.0",
      author="Alejandro Rivas Garcia"
      )