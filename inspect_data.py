from datasets import load_dataset

ds = load_dataset("nkirschi/oxford-flowers")
print(ds)
print(ds["train"].features)
print(ds["train"][0])