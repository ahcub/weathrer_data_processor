rmdir /Q /S build
rmdir /Q /S dist
C:\Users\Alex\Miniconda\envs\python2\Scripts\pyinstaller.exe --onefile --hidden-import packaging --hidden-import packaging.version --hidden-import packaging.specifiers --hidden-import packaging.requirements generate_weather_report.py
copy paths.cfg dist\paths.cfg
