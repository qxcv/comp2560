# Project code

To install on Ubuntu, do:

    $ sudo apt-get install -y python3-virtualenv libjpeg-dev
    $ virtualenv env -p `which python3`
    $ . env/bin/activate
    $ pip install -r requirements.txt
    $ py.test

If the tests pass, you're good to go! Just remember to keep the same shell open,
or to re-run `. env/bin/activate` if you need to open another shell.
