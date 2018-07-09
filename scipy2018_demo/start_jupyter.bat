call demo_config.bat
start http://localhost:8888?token=Default
%SSH% -L 8888:localhost:8888 %HOST% bash -c "pwd;cd %REMOTE_DIR%;%REMOTE_ENV%;exec ./start_jupyter.sh"
