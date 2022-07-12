# InteractiveInference

`IFE.ipynb` is from https://github.com/vschaik/Active-Inference 
based on 
https://doi.org/10.48550/arXiv.1503.04187


### Building the doc

#### Initialization

* From `docs`, run:
    

    $ sphinx-quickstart 


For `> Separate source and build directories (y/n) [n]: y
`, choose yes ( `y`)

* In `docs/source/conf.py`, change the theme to `sphinx_rtd_theme`. 
It should look like:


    html_theme = 'sphinx_rtd_theme'

* In `docs/source/conf.py`, add the extension to `sphinx.ext.mathjx`. 
It should look like:


    extensions = [
        'sphinx.ext.autodoc',
        'sphinx.ext.napoleon',
        'sphinx.ext.mathjax'
    ]

Note: not sure that the import of autodoc is necessary, but it causes no visible harm.


* Edit the path of the source folder. Uncomment and edit so it looks like this:


    import os
    import sys
    sys.path.insert(0, os.path.abspath('../../src'))

#### Update

* From `docs`, run:  

    
    $ sphinx-apidoc -f -o ../docs/source/ ../src/ 

* Remove the file `docs/source/modules.rtf`


* In `docs/source/index.rst`, add the name of the modules. Example:


    .. toctree::
       :maxdepth: 2
       :caption: Contents:
    
       scenario1
       scenario2


* From `docs`, run:

    
    $ make clean
    $ make html


#### For math

Example

    r"""
        .. math::
            1+2 = 4
    """

Remember the `r` before the triple quote, 
for not strange errors to popup as the content would try to be interpreted. 