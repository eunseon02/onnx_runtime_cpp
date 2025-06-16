# convert pre-trained loftr pytorch weights to onnx format

---

## dependencies

---



```bash
git submodule update --init --recursive

python3 -m pip install -r requirements.txt
```

## :running: how to run

---

- download [ASpanFormer](https://github.com/apple/ml-aspanformer) weights weights_aspanformer.tar from [HERE](https://drive.google.com/file/d/1eavM9dTkw9nbc-JqlVVfGPU5UvTTfc6k/view)

- Extract weights_aspanformer.tar by

```
tar -xvf weights_aspanformer.tar
```


- export onnx weights


```
python3 convert_to_onnx.py --model_path /path/to/weight/outdoor.ckpt
```


## Note

