import subprocess

edp_file_path = "meshWithExpReal.edp"
bashCommand = ["Freefem++",  edp_file_path, "0.1", "1"]
process = subprocess.Popen(bashCommand, stdout=subprocess.PIPE)
#output, error = process.communicate()
#print(output)