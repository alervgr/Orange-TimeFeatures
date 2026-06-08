from setuptools import setup

with open('README.md', 'r', encoding='utf-8') as f:
    ABOUT = f.read()

CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'Environment :: Plugins',
    'Programming Language :: Python',
    'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
    'Operating System :: OS Independent',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'Intended Audience :: Developers',
]

setup(name="TimeFeatures",
      packages=["timefeatures.widgets"],
      package_data={"timefeatures.widgets": ["icons/*.svg", "icons/*.png"]},
      entry_points={"orange.widgets": "Time-Features = timefeatures.widgets"},
      install_requires=[
          "numpy>=1.22.4",
          "AnyQt>=0.2.0",
          "PyQt5>=5.15.6",
          "scipy>=1.7.3",
          "psycopg2-binary>=2.9.9",
          "Orange3-Network>=1.8.0",
      ],
      version="1.0.18",
      author="Alejandro Rivas García",
      author_email="alejandrorivasgarcia@gmail.com",
      keywords=[
          'orange3 add-on', 'timefeatures', 'graph', 'time series', 'data mining', 'graph visualization', 'orange',
          'addon', 'synthetic data'
      ],
      url="https://github.com/alervgr/Orange-TimeFeatures",
      license="GPL3+",
      long_description=ABOUT,
      long_description_content_type='text/markdown',
      description="Timefeatures add-on for Orange 3 data mining software.",
      classifiers=CLASSIFIERS,
      )
