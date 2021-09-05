I recommend that you start with the gps, than the User_Camera (drowsiness decector), and than the distance measurements. This is to ensure that you will have the neceassry libaries needed in the virtual environment





# Pi

**Update the RPI (try to update the RPI once a day)**




(updating can take a little as a minute, and sometimes up to one hour)







sudo apt-get update

sudo apt-get upgrade 





**Create a Virtual ENV**




(virtual env is recommended as it helps keeps packages organzied. Some programs/code require certain version of libraries that may not be comptaible with other the libraires installed, so to avoid any issues it is best to use a virtual env.)






sudo pip3 install virtualenv

python3 -m venv name-env

source name-env/bin/activate

deactivate ( to leave virtual env)

