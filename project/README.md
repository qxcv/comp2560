# Project code

To install on Ubuntu, do:

    $ sudo apt-get install -y python-virtualenv libjpeg-dev
    $ virtualenv env -p `which python2`
    $ . env/bin/activate
    (env)$ pip install -r requirements.txt

I've used `(env)` to denote a shell in which the required virtual environment
has been activated. Remember that you'll need to activate the virtual
environment in `env/` (using `. env/bin/activate`) each time you open a *new*
shell in which you wish to run the project code.

The above steps will install most of the required dependences, but *not*
pycaffe. To install pycaffe, you should build Caffe as appropriate for your
environment (use `make distribute` once you have an appropriate
`Makefile.config`), then copy the produced Python package (the entire `caffe`
subdirectory in Caffe's `distribute/python` directory) into
`env/lib/python2.7/site-packages`.

Once you have Caffe installed, you can run tests with:

    (env)$ py.test

If the tests pass, you're good to go! Try running `train.py` to train and test a
model on the LSP data.
