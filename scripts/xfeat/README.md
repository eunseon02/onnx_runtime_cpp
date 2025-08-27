# convert pre-trained xfeat pytorch weights to onnx format

---

## dependencies

---

- pytorch 

-

```bash
```

## :running: how to run

---

- update submodule

```bash
git submodule update --init --recursive
```

- export onnx weights

```
cd XFeat && python3 export.py --xfeat_path ../xfeat.pt --top_k 4096 --dynamic
```
