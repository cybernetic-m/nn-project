import os

training_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../training'))
save_path = os.path.join(training_path, "results")
print(save_path)


